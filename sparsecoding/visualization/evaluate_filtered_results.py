import numpy as np
import scipy.optimize
import scipy.stats as st
import sys

import utils

relu_threshold = 0.005

samples = None

def evaluate_match(true_path, output_path):
    # Read ground truth matrix.
    N = 0
    K = 0
    with open(true_path) as matrix_file:
        file_data = matrix_file.readlines()
        first_line_integers = [int(x) for x in file_data[0].split()]
        N = first_line_integers[0]
        K = first_line_integers[1]
        A = np.zeros((N, K))
        for i in range(1, len(file_data)):
            A[i - 1] = np.fromstring(file_data[i], sep=' ')

    with open(output_path) as matrix_file:
        file_data = matrix_file.readlines()
        first_line_integers = [int(x) for x in file_data[0].split()]
        N_out = first_line_integers[0]
        K_out = first_line_integers[1]
        B = np.zeros((N_out, K_out))
        for i in range(1, len(file_data)):
            B[i - 1] = np.fromstring(file_data[i], sep=' ')

    # Filter B.
    pseudoinverse = np.linalg.pinv(B)
    h = np.matmul(samples, np.transpose(pseudoinverse))
    h = np.vectorize(utils.get_ReLU(relu_threshold))(h)
    p = np.sum(np.greater(h, 0.0).astype(float), 0) / samples.shape[0]

    filter_indices = np.nonzero(np.greater(p, 0.0))[0]
    p_filtered = p[filter_indices]
    B_filtered = B[:, filter_indices]

    match_threshold = 2 * 1e-3

    # If duplicates, keep one with largest prior.
    # p_sorted = np.flip(np.argsort(p_filtered), 0)
    # new_filter = []
    # for i in range(p_filtered.shape[0]):
    #     duplicate = False
    #     # Check if duplicate.
    #     for j in range(0, i):
    #         if (np.linalg.norm(B[:, p_sorted[i]] - B[:, p_sorted[j]]) ** 2) <= match_threshold:
    #             duplicate = True
    #             break
    #     if not duplicate:
    #         new_filter.append(p_sorted[i])
    #
    # p_filtered = p_filtered[new_filter]
    # B_filtered = B_filtered[:, new_filter]

    match_cost = np.zeros((A.shape[1], B_filtered.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B_filtered.shape[1]):
            match_cost[i, j] = np.linalg.norm(A[:, i] - B_filtered[:, j]) ** 2
    match_A, match_B = scipy.optimize.linear_sum_assignment(match_cost)

    match_error = 0.0

    matched = 0
    matched_B = np.zeros(B.shape[1])
    for i in range(len(match_A)):
        match_error += match_cost[match_A[i], match_B[i]]
        if match_cost[match_A[i], match_B[i]] < match_threshold:
            matched_B[match_B[i]] = 1
            matched += 1

    return match_error, len(p_filtered), matched

def evaluate_filtered_match_iterative(true_path, base_output_path, appendix_output_path, runs):
    matches = []
    errors = []
    for i in range(1, runs + 1):
        output_path = base_output_path + '/R' + str(i) + '/' + appendix_output_path
        error, filtered, matched = evaluate_match(true_path, output_path)
        errors.append(error)
        matches.append((filtered, matched))
    return matches, errors

def main():
    true_path = sys.argv[1]
    samples_path = sys.argv[2]
    base_output_path = sys.argv[3]
    appendix_output_path = sys.argv[4]
    runs = int(sys.argv[5])

    global samples
    samples = utils.read_samples(samples_path)
    samples = np.copy(samples[1000 :])

    matches, errors = evaluate_filtered_match_iterative(true_path, base_output_path, appendix_output_path, runs)
    print 'Filtered matches:', matches
    print 'Perfect:', np.sum([(x[0] == 24 and x[1] == 24) for x in matches])
    print 'Errors:', errors, np.average(errors)

if __name__ == "__main__":
   main()
