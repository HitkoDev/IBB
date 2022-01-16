#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json

import cv2
import tensorpack.utils.viz as tpviz
from pycocotools import mask as cocomask

from config import config as cfg
from viz import draw_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation.', required=True)
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py", nargs='+')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    with open(args.load, 'r') as f:
        predictions = json.load(f)

    for p in predictions:
        path = '{}/train/{}'.format(cfg.DATA.BASEDIR, p['image_id'])
        img = cv2.imread(path)
        mask = cocomask.decode(p["segmentation"])
        i2 = draw_mask(img, mask)
        tpviz.interactive_imshow(i2)
