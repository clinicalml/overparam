import numpy as np

import os
import sys

''' Generates the model parameters by filtering the basefile.txt file, which
    is the result of learning a noisy-OR network model from the real-world
    plants datatset.'''


def read_parameters(model_path):
    infile = file(model_path)

    line = infile.readline()
    dim_latent, dim_observed = [int(x) for x in line.split()]

    line = infile.readline()
    p = np.array([float(x) for x in line.split()])
    f = np.zeros((dim_latent, dim_observed))
    for i in range(dim_latent):
        line = infile.readline()
        f[i] = np.array([float(x) for x in line.split()])
    line = infile.readline()
    n = np.array([float(x) for x in line.split()])

    return p, f, n

p, f, n = read_parameters('basefile.txt')

ban = np.zeros(p.shape[0], dtype=int)
ban_threshold = 0.01 # remove all latent variables with priors less than this
banned = 0
f_threshold = 0.50 # make 1 everything larger than this

for i in range(p.shape[0]):
    if p[i] <= ban_threshold:
        ban[i] = 1
        banned += 1

for i in range(f.shape[0]):
    for j in range(f.shape[1]):
        if f[i, j] >= f_threshold:
            f[i, j] = 1.0
    if np.amin(f[i]) == 1.0 and ban[i] == 0:
        ban[i] = 1
        banned += 1

# for i in range(f.shape[0]):
#     if not ban[i]:
#         for j in range(i + 1, f.shape[0]):
#             if not ban[j]:
#                 print np.sum(np.abs(f[i] - f[j]))

priors_file = open('priors.txt', 'w')
weights_file = open('weights.txt', 'w')
noise_file = open('noise.txt', 'w')

priors_file.write(str(p.shape[0] - banned) + '\n')
for i in range(p.shape[0]):
    if not ban[i]:
        priors_file.write(str(p[i]) + '\n')

for i in range(f.shape[0]):
    if not ban[i]:
        for j in range(f.shape[1]):
            weights_file.write(str(f[i, j]) + ' ')
        weights_file.write('\n')

noise_file.write(str(n.shape[0]) + '\n')
for i in range(n.shape[0]):
    noise_file.write(str(n[i]) + '\n')

common_file = open('common.txt', 'w')
common_file.write(str(p.shape[0] - banned) + ' ' + str(n.shape[0]) + '\n')
for i in range(p.shape[0]):
    if not ban[i]:
        common_file.write(str(p[i]) + ' ')
common_file.write('\n')
for i in range(f.shape[0]):
    if not ban[i]:
        for j in range(f.shape[1]):
            common_file.write(str(f[i, j]) + ' ')
        common_file.write('\n')
for i in range(n.shape[0]):
    common_file.write(str(n[i]) + ' ')
common_file.write('\n')
common_file.close()
