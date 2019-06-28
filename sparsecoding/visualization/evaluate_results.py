import numpy as np
import scipy.optimize
import scipy.stats as st
import sys

import utils

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

    match_cost = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            match_cost[i, j] = np.linalg.norm(A[:, i] - B[:, j]) ** 2
    match_A, match_B = scipy.optimize.linear_sum_assignment(match_cost)

    match_error = 0.0
    match_threshold = 2 * 1e-3

    matched = 0
    matched_B = np.zeros(B.shape[1])
    for i in range(len(match_A)):
        match_error += match_cost[match_A[i], match_B[i]]
        if match_cost[match_A[i], match_B[i]] < match_threshold:
            matched_B[match_B[i]] = 1
            matched += 1

    return match_error, matched

def evaluate_match_iterative(true_path, base_output_path, appendix_output_path, runs):
    matches = []
    errors = []
    for i in range(1, runs + 1):
        output_path = base_output_path + '/R' + str(i) + '/' + appendix_output_path
        error, matched = evaluate_match(true_path, output_path)
        errors.append(error)
        matches.append(matched)
    return matches, errors

def main():
    true_path = sys.argv[1]
    base_output_path = sys.argv[2]
    appendix_output_path = sys.argv[3]
    runs = int(sys.argv[4])

    matches, errors = evaluate_match_iterative(true_path, base_output_path, appendix_output_path, runs)

    runs = len(matches)
    print 'Matches:', matches
    print 'Perfect:', np.sum([(x == 24) for x in matches])
    print 'Errors:', errors
#     print 'Matches:', np.average(matches), '+-' + str(st.t.ppf(1.0 - (0.05 / 2.0), runs - 1) * st.sem(matches))
#     print 'Perfect:', np.sum([(x == 24) for x in matches]), np.average([(x == 24) for x in matches]), '+-' + str(st.t.ppf(1.0 - (0.05 / 2.0), runs - 1) * st.sem([(x == 24) for x in matches]))
#     print 'Errors:', np.average(errors), '+-' + str(st.t.ppf(1.0 - (0.05 / 2.0), runs - 1) * st.sem(errors))

if __name__ == "__main__":
   main()
