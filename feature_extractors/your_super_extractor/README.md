# Preparing the data

```sh
python prepare-detections.py --regions out.json --dataset ../../data/awe
```

This will extract ears and masks from images based on regions detected in the detection phase, and place them under `../../data/recognition`.
