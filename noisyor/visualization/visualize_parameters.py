import os
import sys

import numpy as np
from read_parameters import read_parameters


model_path = sys.argv[1]
p, f, n = read_parameters(model_path)

L = p.shape[0]
N = n.shape[0]

print 'Priors:'
for i in range(L):
    print p[i],
print '\n'

print 'Weights:'
for i in range(L):
    for j in range(N):
        print f[i, j],
    print '\n'

print 'Noise:'
for i in range(N):
    print n[i],
print '\n'
