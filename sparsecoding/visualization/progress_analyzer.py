import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import sys

# matrix_name = name of main directory, used with the generators
# results_name = name of results directory, used with the trainer
# iter_freq = try files of the form "iter_[iter_freq * x].txt"
# iter_max = the maximum file is "iter_[iter_max].txt"
if len(sys.argv) < 4:
    print "usage: python compare_results.py matrix_name results_name iter_freq iter_max"
    sys.exit()
matrix_name = sys.argv[1]
results_name = sys.argv[2]
iter_freq = int(sys.argv[3])
iter_max = int(sys.argv[4])

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

iters = []
errors = []
matches = []

matches_B = []
for it in range(iter_freq, iter_max, iter_freq):
    filename = matrix_name + '/results/' + results_name + '/iter_' + str(it) + '.txt'
    B = np.loadtxt(filename)

    match_cost = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            match_cost[i, j] = np.linalg.norm(A[:, i] - B[:, j])
    match_A, match_B = scipy.optimize.linear_sum_assignment(match_cost)

    iters.append(it)
    errors.append(np.sum(match_cost[match_A, match_B]))
    matches_B.append(match_B)

for i in range(len(matches_B)):
    matches.append(np.sum(matches_B[-1] == matches_B[i]))

plt.plot(iters, errors, 'r-')
plt.plot(iters, matches, 'b-')
plt.legend(('error in A', 'correct matches'))
plt.xlabel('iterations')
# plt.xticks(v_iter)
plt.show()
