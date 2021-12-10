import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
from pathlib import Path

import os
from glob import glob
from pycocotools import mask as cocomask

import cv2

__all__ = ["register_awe"]

class AWEDataset(DatasetSplit):
    def __init__(self, base_dir, split):
        """Load a subset of the Ear dataset.
        base_dir: Root directory of the dataset.
        split: Subset to load: train or val
        """

        self.images = []
        self.imagesDict = {}

        # Train or validation dataset?
        assert split in ["train", "val"]
        src_dir = os.path.join(base_dir, split)
        annot_dir = os.path.join(base_dir, '{}annot'.format(split))
        bb_dir = os.path.join(base_dir, '{}annot_rect'.format(split))
        mask_dir = os.path.join(base_dir, '{}_masks'.format(split))
        images = glob('{}/**/*'.format(src_dir), recursive=True)
        images = [i for i in images if os.path.exists(i.replace(src_dir, annot_dir)) and os.path.exists(i.replace(src_dir, bb_dir))]

        for path in images:
            img = cv2.imread(path)
            mask = cv2.imread(path.replace(src_dir, annot_dir))
            bb = cv2.imread(path.replace(src_dir, bb_dir))
            w, h, c = img.shape

            gray = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            mask_path = path.replace(src_dir, mask_dir) + '.npy'
            masks = []

            if not os.path.exists(mask_path):
                if not os.path.exists(os.path.dirname(mask_path)):
                    os.makedirs(os.path.dirname(mask_path))

                i = 0
                for c in contours:
                    m = np.full((w, h), False)
                    x1 = min([p[0][1] for p in c if p[0][1] >= 0])
                    x2 = max([p[0][1] for p in c if p[0][1] >= 0]) + 1
                    y1 = min([p[0][0] for p in c if p[0][0] >= 0])
                    y2 = max([p[0][0] for p in c if p[0][0] >= 0]) + 1
                    s = mask[x1:x2, y1:y2, 1] > 0
                    m[x1:x2, y1:y2] = s
                    mask_path = path.replace(src_dir, mask_dir) + '_' + str(i) + '.npy'
                    with open(mask_path, 'wb+') as file:
                        np.save(file, m)
                    masks.append(mask_path)
                    i += 1
            else:
                i = 0
                for c in contours:
                    mask_path = path.replace(src_dir, mask_dir) + '_' + str(i) + '.npy'
                    masks.append(mask_path)
                    i += 1

            bbox = []
            for c in contours:
                y1 = min([p[0][1] for p in c if p[0][1] >= 0])
                y2 = max([p[0][1] for p in c if p[0][1] >= 0]) + 1
                x1 = min([p[0][0] for p in c if p[0][0] >= 0])
                x2 = max([p[0][0] for p in c if p[0][0] >= 0]) + 1
                bbox.append([x1, y1, x2, y2])

            img = dict(
                image_id=path.replace(src_dir, '')[1:],
                path=str(Path(path).absolute()),
                masks=[str(Path(m).absolute()) for m in masks],
                objects=len(contours),
                bbox=np.array(bbox).astype(np.float32),
                width=w, 
                height=h
            )

            m = []
            for p in img['masks']:
                with open(p, 'rb') as file:
                    mask = np.load(file)
                    m.append(cocomask.encode(np.asfortranarray(mask)))

            img['rle'] = m

            self.images.append(img)
            self.imagesDict[img['image_id']] = img


    def training_roidbs(self):
        """
        Returns:
            roidbs (list[dict]):

        Produce "roidbs" as a list of dict, each dict corresponds to one image with k>=0 instances.
        and the following keys are expected for training:

        file_name: str, full path to the image
        boxes: numpy array of kx4 floats, each row is [x1, y1, x2, y2]
        class: numpy array of k integers, in the range of [1, #categories], NOT [0, #categories)
        is_crowd: k booleans. Use k False if you don't know what it means.
        segmentation: k lists of numpy arrays.
            Each list of numpy arrays corresponds to the mask for one instance.
            Each numpy array in the list is a polygon of shape Nx2,
            because one mask can be represented by N polygons.
            Each row in the Nx2 array is a (x, y) coordinate.

            If your segmentation annotations are originally masks rather than polygons,
            either convert it, or the augmentation will need to be changed or skipped accordingly.

            Include this field only if training Mask R-CNN.

        Coordinates in boxes & polygons are absolute coordinates in unit of pixels, unless
        cfg.DATA.ABSOLUTE_COORD is False.
        """
        return [{
            "file_name": i['path'],
            "image_id": i["image_id"],
            "boxes":i['bbox'],
            "class":np.array([1 for j in range(i['objects'])]),
            "is_crowd":np.array([False for j in range(i['objects'])]),
            "segmentation": i["masks"],
            "rle": i['rle']
         } for i in self.images]

    def inference_roidbs(self):
        """
        Returns:
            roidbs (list[dict]):

            Each dict corresponds to one image to run inference on. The
            following keys in the dict are expected:

            file_name (str): full path to the image
            image_id (str): an id for the image. The inference results will be stored with this id.
        """

        return [{
            "file_name": i['path'],
            "image_id": i["image_id"],
            "boxes":i['bbox'],
            "class":np.array([1 for j in range(i['objects'])]),
            "is_crowd":np.array([False for j in range(i['objects'])]),
            "segmentation": i["masks"],
            "rle": i['rle']
         } for i in self.images]

    def eval_inference_results(self, results, output=None):
        """
        Args:
            results (list[dict]): the inference results as dicts.
                Each dict corresponds to one __instance__. It contains the following keys:

                image_id (str): the id that matches `inference_roidbs`.
                category_id (int): the category prediction, in range [1, #category]
                bbox (list[float]): x1, y1, x2, y2
                score (float):
                segmentation: the segmentation mask in COCO's rle format.
            output (str): the output file or directory to optionally save the results to.

        Returns:
            dict: the evaluation results.
        """

        n = []
        p = []
        r = []
        for res in results:
            img = self.imagesDict[res['image_id']]

            rle = img['rle'][0]
            m_iou = 0.
            for m in img['rle']:
                iou = cocomask.iou([m], [res["segmentation"]], [False])
                iou = np.array(iou).flatten()[0]
                if iou > m_iou:
                    rle = m

            iou = cocomask.iou([rle], [res["segmentation"]], [False])
            iou = np.array(iou).flatten()[0]
            n.append(iou)

            tp = cocomask.merge([rle, res["segmentation"]], True)
            tpfn = rle
            tpfp = res["segmentation"]

            prec = cocomask.area(tp)/cocomask.area(tpfp)
            p.append(prec)

            rec = cocomask.area(tp)/cocomask.area(tpfn)
            r.append(rec)

            res['iou'] = iou
            res['precision'] = prec
            res['recall'] = rec

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)

        n = np.array(n).flatten()
        p = np.array(p).flatten()
        r = np.array(r).flatten()

        return dict(
            iou_mean=np.mean(n),
            iou_std=np.std(n),
            iou_var=np.var(n),
            precision__mean=np.mean(p),
            precision__std=np.std(p),
            precision__var=np.var(p),
            recall_mean=np.mean(r),
            recall_std=np.std(r),
            recall_var=np.var(r)
        )


def register_awe(basedir):
    for split in ["train", "val"]:
        name = "awe_" + split
        DatasetRegistry.register(name, lambda x=split: AWEDataset(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "ear"])


if __name__ == '__main__':
    basedir = '../../data/awe'
    ds = AWEDataset(basedir, "train")
    roidbs = ds.training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
        imshow(vis)
