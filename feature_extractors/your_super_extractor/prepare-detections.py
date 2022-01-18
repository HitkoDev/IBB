#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
from pycocotools import mask as cocomask
from tensorpack.utils.palette import PALETTE_RGB


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    color = np.asarray(color, dtype=np.float32)
    print(np.shape(mask))
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', help='load regions', required=True)
    parser.add_argument('--dataset', help='dataset path', required=True)

    args = parser.parse_args()

    with open(args.regions, 'r') as f:
        regions = json.load(f)
        regions = [r for r in regions if r["score"] > 0.65]

    tr = pd.read_csv('awe-translation.csv')
    maping = {}
    for i, r in tr.iterrows():
        maping[r['Detection filename'].replace('test', 'val')] = r['Class ID']

    for p in regions:
        path = '{}/{}'.format(args.dataset, p['image_id'])
        img = cv2.imread(path)
        mask = cocomask.decode(p["segmentation"])
        # Take some region around the mask
        mask = scipy.ndimage.binary_dilation(mask, iterations=3)
        i2 = img * np.stack([mask, mask, mask], axis=2)
        cs = maping[p['image_id']]
        path = p['image_id'].replace('/', '/{}/'.format(cs))
        path = '{}/{}'.format(args.dataset.replace('awe', 'recognition'), path)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        cv2.imwrite(path, i2)
        mask_path = re.sub(r'\.[^\.]*$', '.npy', path)
        with open(mask_path, 'wb+') as file:
            np.save(file, mask)

        path = p['image_id'].replace('/', '/{}/'.format(cs))
        path = '{}/{}'.format(args.dataset.replace('awe', 'recognition2'), path)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        cv2.imwrite(path, img)
        mask_path = re.sub(r'\.[^\.]*$', '.npy', path)
        with open(mask_path, 'wb+') as file:
            np.save(file, mask)
