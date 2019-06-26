import os
import sys

import numpy as np
from read_parameters import read_parameters
from evaluate_filtered_results import filter_parameters


label = "ab ak al ar az bc ca co ct dc de dengl fl fraspm ga gl hi ia id il in ks ky la lb ma mb md me mi mn mo ms mt nb nc nd ne nf nh nj nm ns nt nu nv ny oh ok on or pa pe pr qc ri sc sd sk tn tx ut va vi vt wa wi wv wy yt".split()

model_path = sys.argv[1]
p, f, n = read_parameters(model_path)
# p, f = filter_parameters(p, f)

L = p.shape[0]
N = n.shape[0]

print 'Priors:'
for i in range(L):
    print p[i],
print '\n'

# print 'Weights:'
# for i in range(L):
#     for j in range(N):
#         print f[i, j],
#     print '\n'

print 'Interpretation (with filtering):'
threshold = 0.98
for i in range(L):
    if p[i] >= 0.01 and p[i] <= 0.99:
        for j in range(N):
            if f[i, j] <= threshold:
                print label[j],
        print '\n'

print 'In none:'
for j in range(N):
    any = False
    for i in range(L):
        if f[i, j] <= threshold:
            any = True
    if not any:
        print label[j],
print '\n'

# print 'Noise:'
# for i in range(N):
#     print n[i],
# print '\n'
