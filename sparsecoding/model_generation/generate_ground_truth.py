import numpy as np
import os
import sys

''' Generates a sparse coding ground truth matrix. '''


# Read sys.argv
# Generates the matrix in file matrix_name/true_params/matrix.txt
if len(sys.argv) < 2:
    print 'usage: python generate_ground_truth.py matrix_name'
    sys.exit()
matrix_name = sys.argv[1]

# Creating the matrix
def generate_uniform(N, K):
    A = np.random.uniform(size=(N, K))
    for i in range(A.shape[1]):
        A[:, i] /= np.sum(A[:, i])
    return A

def generate_gaussian_unit(N, K):
    A = np.random.normal(size=(N, K))
    for i in range(A.shape[1]):
        A[:, i] /= np.linalg.norm(A[:, i])
    return A

def generate_correlated(N, K_half, angle):
    A = np.zeros((N, 2 * K_half))
    for i in range(0, A.shape[1], 2):
        Ap = np.random.normal(0.0, 1.0, size=(N))
        Bp = np.random.normal(0.0, 1.0, size=(N))

        Ap /= np.linalg.norm(Ap)
        Bp -= np.dot(Ap, Bp) * Ap
        Bp /= np.linalg.norm(Bp)
        Cp = np.linalg.norm(Ap) * (np.cos(angle) * Ap + np.sin(angle) * Bp)

        A[:, i] = Ap
        A[:, i + 1] = Cp / np.linalg.norm(Cp)

        # print 'Dot product:', np.dot(A[:, i], A[:, i + 1])
    return A

# A = generate_gaussian_unit(100, 24)
A = generate_correlated(100, 12, 10.0 * (2 * np.pi / 360.0)) # A.shape = (100, 24)

# Printing the matrix
os.makedirs(matrix_name + '/true_params')
# os.makedirs(matrix_name + '/samples')
# os.makedirs(matrix_name + '/runs')
matrix_file = file(matrix_name + '/true_params/matrix.txt', 'w')

print >>matrix_file, A.shape[0], A.shape[1]
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        print >>matrix_file, A[i, j],
    print >>matrix_file, ''

matrix_file.close()
