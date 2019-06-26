import numpy as np
import os
import sys

def read_samples(samples_filename, samples=None):
    A = np.loadtxt(samples_filename)
    if samples:
        A = A[0:samples, :]
    return A

def get_initial_A_uniform(N, K):
    A = np.random.uniform(size=(N, K))
    for i in range(A.shape[1]):
        A[:, i] /= np.sum(A[:, i])
    return A

def get_initial_A_gaussian(N, K):
    A = np.random.normal(size=(N, K))
    for i in range(A.shape[1]):
        A[:, i] /= np.linalg.norm(A[:, i])
    return A
