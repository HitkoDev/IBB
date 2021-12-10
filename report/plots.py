import json

import numpy as np
import matplotlib.pyplot as plt

with open('summary.json', 'r') as f:
    data = json.load(f)

# tpr

tpr = np.array([v["tp"] / 287 for v in data])
fpr = np.array([v["fp"] for v in data])

x = [fpr[0]]
y = [tpr[0]]
p = fpr[0]
for i in range(len(fpr)):
    if p != fpr[i]:
        p = fpr[i]
        x.append(p)
        y.append(tpr[i])


fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('False positives')
ax.set_ylabel('True positive rate')
ax.legend()
ax.grid()
plt.show()

# metrics

x = np.array([v["minScore"] for v in data])
iou = np.array([v["iou"] for v in data])
ioue = np.array([v["iouS"] for v in data])
precision = np.array([v["precision"] for v in data])
precisione = np.array([v["precisionS"] for v in data])
recall = np.array([v["recall"] for v in data])
recalle = np.array([v["recallS"] for v in data])

fig, ax = plt.subplots()
r, = ax.plot(x, iou, label='IoU')
ax.plot(x, iou - ioue, linestyle='dashed', c=r.get_color(), alpha=0.5)
ax.plot(x, iou + ioue, linestyle='dashed', c=r.get_color(), alpha=0.5)
r, = ax.plot(x, precision, label='precision')
ax.plot(x, precision - precisione, linestyle='dashed', c=r.get_color(), alpha=0.5)
ax.plot(x, precision + precisione, linestyle='dashed', c=r.get_color(), alpha=0.5)
r, = ax.plot(x, recall, label='recall')
ax.plot(x, recall - recalle, linestyle='dashed', c=r.get_color(), alpha=0.5)
ax.plot(x, recall + recalle, linestyle='dashed', c=r.get_color(), alpha=0.5)
ax.set_xlabel('Minimal acceptance score')
ax.set_ylabel('Metric value')
ax.plot(x, tpr, label='true positive rate')
r, = ax.plot(x, fpr/287, label='false positives')

ax2 = ax.twinx()
ax2.plot(x, fpr, label='False positives', color=r.get_color())
ax2.set_ylabel('False positives')

ax.legend(loc='lower left')
r.set_linewidth(0)
ax.grid()
plt.show()