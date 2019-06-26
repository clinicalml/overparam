# visualize_images.py
# a script to visualize learned structure in the synthetic images dataset

import sys
from read_parameters import read_parameters
from itertools import izip_longest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20

model_path = sys.argv[1]
p, f, n = read_parameters(model_path)

L = f.shape[0]
images_temp = []
for i in range(L):
    images_temp.append(np.hstack([1 - f[i, :].reshape((8, 8)), np.zeros((8, 1))]))

priors, images = zip(*sorted(izip_longest(p, images_temp, fillvalue=0), reverse=True, key=lambda x: x[0]))

images = np.vstack([np.hstack(images), np.hstack([np.ones(images[-1].shape) * p for p in priors])])

plt.figure(str(L), figsize=(L + 2, 2))
plt.imshow(
    np.vstack(images), interpolation="nearest", cmap=plt.cm.Greys_r)
plt.xticks([], [])
plt.yticks([], [])

plt.show()
