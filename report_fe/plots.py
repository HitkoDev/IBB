import json

import numpy as np
import matplotlib.pyplot as plt

with open('out.json', 'r') as f:
    data = json.load(f)

# tpr

tpr = np.array([v["tp"] / 247 for v in data])
fpr = np.array([v["fp"] / 247 for v in data])

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