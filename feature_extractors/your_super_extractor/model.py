import math
import os
import random
from glob import glob

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from skimage.io import imread
from skimage.transform import resize

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SIZE = 226

seq = iaa.Sequential([
    iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 0.2)),       # sharpen the image
    iaa.Affine(rotate=(-15, 15)),  # rotate by -45 to 45 degrees (affects segmaps)
    iaa.flip.Fliplr(0.5)
], random_order=True)


class Dataset(object):
    def __init__(self, base_dir, split) -> None:
        super().__init__()
        src_dir = os.path.join(base_dir, split)
        images = glob('{}/**/*.png'.format(src_dir), recursive=True)

        self.images = []

        cl = set()

        for i in images:
            i = i.replace('\\', '/')
            c = int(i.split('/')[-2])
            cl.add(c)
            m = i.replace('.png', '.npy')
            with open(m, 'rb') as file:
                mask = np.load(file)
            if(np.sum(mask) > 100):
                self.images.append({
                    'image_id': i,
                    'class': c,
                    'mask': m
                })

        self.classes = [c for c in cl]
        self.count_classes = max(self.classes) + 1
        e = np.eye(self.count_classes)
        for i in self.images:
            i['label'] = e[i['class']]

        self.imagesDict = {k: [i for i in self.images if i['class'] == k] for k in self.classes}


class AWESequence(tf.keras.utils.Sequence):

    def __init__(self, ds, batch_size, aug=False):
        self.batch_size = batch_size
        self.ds = ds
        self.aug = aug

    def __len__(self):
        #        return math.ceil(len(self.ds.classes) / self.batch_size)
        return math.ceil(len(self.ds.images) / self.batch_size)

    def __getitem__(self, idx):
        x = []
        y = []

#        for cls in self.ds.classes[idx * self.batch_size:(idx + 1) * self.batch_size]:
#            for img in self.ds.imagesDict[cls]:
        for img in self.ds.images[idx * self.batch_size:(idx + 1) * self.batch_size]:
            image = imread(img['image_id'])
            with open(img['mask'], 'rb') as file:
                mask = np.load(file)

            segmap = SegmentationMapsOnImage(mask.copy(), shape=image.shape)
            image2, mask2 = seq(image=image.copy(), segmentation_maps=segmap)
            mask2 = mask2.get_arr()

            if(self.aug and np.sum(mask2) > 100):
                i, j = np.where(mask2)
                x1 = np.min(i)
                x2 = np.max(i)
                y1 = np.min(j)
                y2 = np.max(j)
                image2 = image2[x1:x2, y1:y2]

                image2 = resize(image2, (SIZE, SIZE))
                image2 = tf.keras.applications.resnet.preprocess_input(image2)

                x.append(image2)
                y.append(img['label'])

            # Skip aug
            i, j = np.where(mask)
            x1 = np.min(i)
            x2 = np.max(i)
            y1 = np.min(j)
            y2 = np.max(j)
            image = image[x1:x2, y1:y2]

            image = resize(image, (SIZE, SIZE))
            image = tf.keras.applications.resnet.preprocess_input(image)

            x.append(image)
            y.append(img['label'])

        return np.array(x), np.array(y)

    def on_epoch_end(self):
        #        random.shuffle(self.ds.classes)
        random.shuffle(self.ds.images)


ds = Dataset('../../data/recognition2', 'train')
ds_val = Dataset('../../data/recognition2', 'val')

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(SIZE, SIZE, 3)
)

inputs = tf.keras.Input(shape=(SIZE, SIZE, 3))
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
top_dropout_rate = 0.2
x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = tf.keras.layers.Dense(
    ds.count_classes, activation='softmax', name="pred")(x)

# First train on images with background (bounding box)

model = tf.keras.Model(inputs, outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./logs2',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_acc',
    mode='max')

model.fit(AWESequence(ds, 15, True), epochs=100, validation_data=AWESequence(ds_val, 15), callbacks=[model_checkpoint_callback])

# Then fine tune on segmented images

model = tf.keras.Model(inputs, outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights('./logs2')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./logs4',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_acc',
    mode='max')


ds = Dataset('../../data/recognition', 'train')
ds_val = Dataset('../../data/recognition', 'val')

model.fit(AWESequence(ds, 15, True), epochs=500, validation_data=AWESequence(ds_val, 15), callbacks=[model_checkpoint_callback])


model = tf.keras.Model(inputs, outputs)
model.load_weights('./logs4')
model.save('model2.pb')
