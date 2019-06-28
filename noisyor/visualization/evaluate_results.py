import os
import sys

import numpy as np
import scipy.stats as st
import scipy.optimize
import matplotlib.pyplot as plt

from read_parameters import *
from utils import *

match_threshold = 1.0


def match_latent_variables(p_true, f_true, n_true, p_out, f_out, n_out, norm='inf'):
    L_true = p_true.shape[0]
    L_out = p_out.shape[0]
    N = n_true.shape[0]

    cost = np.zeros((L_true, L_out))
    for i in range(L_true):
        for j in range(L_out):
            if norm == 1:
                cost[i, j] = match_cost_L1(p_true[i], p_out[j], f_true[i], f_out[j])
            else:
                cost[i, j] = match_cost_Linf(p_true[i], p_out[j], f_true[i], f_out[j])
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    return cost, row_ind, col_ind


# Returns latent parameters that match.
def get_matched_latent_variables(p_true, f_true, n_true, p_out, f_out, n_out):
    cost, row_ind, col_ind = match_latent_variables(p_true, f_true, n_true, p_out, f_out, n_out)

    matched_latent_variables = []
    for i in range(len(row_ind)):
        # print cost[row_ind[i], col_ind[i]],
        if cost[row_ind[i], col_ind[i]] <= match_threshold:
            matched_latent_variables.append((row_ind[i], col_ind[i]))
    # print

    return matched_latent_variables


def evaluate_match(true_path, output_path):
    p_true, f_true, n_true = read_true_parameters(true_path)
    p_out, f_out, n_out = read_parameters(output_path)

    return len(get_matched_latent_variables(p_true, f_true, n_true, p_out, f_out, n_out))


def evaluate_match_iterative(true_path, base_output_path, output_file_frequency, runs1, runs2):
    matches = []
    filtered_matches = []
    ll = []
    for i in range(runs1, runs2 + 1):
        log_path = base_output_path + '/R' + str(i) + '/log.txt'
        best_ll, best_index = read_best_likelihood(log_path, output_file_frequency)
        # best_ll = read_likelihood_epoch(log_path, 1)
        # best_index = 2
        ll.append(best_ll)

        output_path = base_output_path + '/R' + str(i) + '/' + 'model_epoch' + str(best_index + 1) + '.dat'
        matches.append(evaluate_match(true_path, output_path))

        p_true, f_true, n_true = read_true_parameters(true_path)
        p_out, f_out, n_out = read_parameters(output_path)
        p_filtered, f_filtered = filter_parameters(p_out, f_out)

        pre_filtered_latent = [x for x in range(p_out.shape[0]) if (p_out[x] > p_threshold and np.amin(f_out[x]) < f_threshold)]
        filtered_matches.append((len(pre_filtered_latent), len(p_filtered), len(get_matched_latent_variables(p_true, f_true, n_true, p_filtered, f_filtered, n_out))))

        # print best_index,
    # print
    return matches, ll, filtered_matches


def main():
    true_path = sys.argv[1] # path to model/true_params
    base_output_path = sys.argv[2] # path to model/runs/..., such that Ri directories are in that path
    output_file_frequency = int(sys.argv[3]) # select best held-out of many iteration, at the given frequency (i.e. if frequncy 100, look at model_epoch100.dat, model_epoch200.dat, ...)
    runs = int(sys.argv[4]) # number of Ri directories

    true_dim_latent = 8

    matches, ll, filtered_matches = evaluate_match_iterative(true_path, base_output_path, output_file_frequency, 1, runs)

    # Summary.
    runs = len(matches)
    # print 'Matches:', np.average(matches), '+-' + str(st.t.ppf(1.0 - (0.05 / 2.0), runs - 1) * st.sem(matches))
    # print 'Log-likelihood:', np.average(ll), '+-' + str(st.t.ppf(1.0 - (0.05 / 2.0), runs - 1) * st.sem(ll))
    # print 'Perfect:', np.sum([(x == true_dim_latent) for x in matches]), np.average([(x == true_dim_latent) for x in matches]), '+-' + str(st.t.ppf(1.0 - (0.05 / 2.0), runs - 1) * st.sem([(x == true_dim_latent) for x in matches]))
    # print 'Filtered perfect matches:', np.sum([(x[1] == true_dim_latent and x[2] == true_dim_latent) for x in filtered_matches])
    # print 'Extra in perfect:', np.sum([(filtered_matches[x][0] != true_dim_latent and matches[x] == true_dim_latent) for x in range(len(filtered_matches))])

    print 'Matches:', matches
    print 'Perfect:', [(x == true_dim_latent) for x in matches]
    print 'Log-likelihood:', ll


if __name__ == "__main__":
   main()
