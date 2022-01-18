import json
import math
import os
from glob import glob

import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SIZE = 226


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

    def __init__(self, ds, batch_size):
        self.batch_size = batch_size
        self.ds = ds

    def __len__(self):
        return math.ceil(len(self.ds.images) / self.batch_size)

    def __getitem__(self, idx):
        x = []
        y = []

        for img in self.ds.images[idx * self.batch_size:(idx + 1) * self.batch_size]:
            image = imread(img['image_id'])
            with open(img['mask'], 'rb') as file:
                mask = np.load(file)

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


ds = Dataset('../../data/recognition', 'val')

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
outputs = tf.keras.layers.Dense(ds.count_classes, activation='softmax', name="pred")(x)

model = tf.keras.Model(inputs, outputs)
model.load_weights('./logs4')
model.save('model2.pb')

res = model.predict(AWESequence(ds, 15))
data = []
for i in range(len(ds.images)):
    img = ds.images[i]
    data.append({
        'image_id': img['image_id'],
        'class': img['class'],
        'prediction': [float(k) for k in list(res[i])]
    })

with open('pred.json', 'w') as f:
    json.dump(data, f)
