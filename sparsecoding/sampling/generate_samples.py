import numpy as np
import os
import sys

''' Samples from a ground truth matrix A, according to x = Ah, where h is
    random, non-negative, has l0-norm $sparsity, and has l1-norm 1. '''


# Read sys.argv
# Generates N samples in file matrix_name/samples/samples.txt
if len(sys.argv) < 4:
    print "usage: python generate_samples.py matrix_name seed num_samples sparsity"
    sys.exit()
matrix_name = sys.argv[1]
seed = int(sys.argv[2])
num_samples = int(sys.argv[3])
sparsity = int(sys.argv[4])

np.random.seed(seed)

# Read ground truth matrix.
N = 0
K = 0
with open(matrix_name + '/true_params/matrix.txt') as matrix_file:
    file_data = matrix_file.readlines()
    first_line_integers = [int(x) for x in file_data[0].split()]
    N = first_line_integers[0]
    K = first_line_integers[1]
    A = np.zeros((N, K))
    for i in range(1, len(file_data)):
        A[i - 1] = np.fromstring(file_data[i], sep=' ')

# Generating the samples.
# For every sample, it selects sparsity topics, and then allocates
# probability uniformly between them.
samples = np.zeros((num_samples, N))
for i in range(num_samples):
    topics = np.random.choice(K, sparsity, replace=False)
    distribution = np.random.random(sparsity)
    distribution /= np.sum(distribution)
    x = np.zeros(K)
    x[topics] = distribution

    samples[i] = np.dot(A, x)

np.savetxt(matrix_name + '/samples/samples.txt', samples)
