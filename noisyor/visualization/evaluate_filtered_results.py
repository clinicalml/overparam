import os
import sys

import numpy as np
import scipy.optimize

from read_parameters import read_true_parameters, read_parameters, read_likelihood
from utils import *
from evaluate_results import get_matched_latent_variables


def evaluate_filtered_match_iterative(true_path, base_output_path, appendix_output_path, runs):
    matches = []
    ll = [] # ll under full model, as reported in the log file of the algorithm
    for i in range(1, runs + 1):
        output_path = base_output_path + '/R' + str(i) + '/' + appendix_output_path
        p_true, f_true, n_true = read_true_parameters(true_path)
        p_out, f_out, n_out = read_parameters(output_path)

        p_filtered, f_filtered = filter_parameters(p_out, f_out)

        matches.append((len(p_filtered), len(get_matched_latent_variables(p_true, f_true, n_true, p_filtered, f_filtered, n_out))))

        log_path = base_output_path + '/R' + str(i) + '/log.txt'
        ll.append(read_likelihood(log_path))
    return matches, ll


def main():
    true_path = sys.argv[1] # path to model/true_params
    base_output_path = sys.argv[2] # path to model/runs/..., such that Ri directories are in that path
    appendix_output_path = sys.argv[3] # file name inside Ri directory to use; e.g. "model_epoch1000.dat"
    runs = int(sys.argv[4]) # number of Ri directories

    matches, ll = evaluate_filtered_match_iterative(true_path, base_output_path, appendix_output_path, runs)
    print 'Filtered matches:', matches
    print 'Perfect:', np.sum([(x[0] == 8 and x[1] == 8) for x in matches])
    print 'Log-likelihood:', ll, np.average(ll)

if __name__ == "__main__":
   main()
