import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from read_parameters import read_true_parameters, read_parameters, read_likelihood
from evaluate_results import match_latent_variables, get_matched_latent_variables

ll_scale = 1.0

true_path = sys.argv[1]
output_path = sys.argv[2] # path to directory with model_epoch files
steps = int(sys.argv[3])

p_true, f_true, n_true = read_true_parameters(true_path)

iters = []
matched_latent_variables = []
ll = []
for epoch in range(1, steps + 1):
    # Get likelihood.
    ll_file = file(output_path + '/log.txt')
    ll_lines = ll_file.readlines()
    ll_lines = [x.strip() for x in ll_lines]
    ll_epoch_index = ll_lines.index('Epoch ' + str(epoch - 1))

    ll_index = -1
    for j in range(ll_epoch_index, len(ll_lines)):
        if ll_lines[j].startswith('Log likelihood'):
            ll_index = j
            break
    ll_value = -float(ll_lines[ll_index].strip().split()[-1]) * ll_scale

    # Get model.
    for i in range(0, 9000, 20 * 10):
        iters.append(epoch + 1.0 * i / 9000)

        path = output_path + '/model_epoch' + str(epoch) + '_iter' + str(i + 1) + '.dat'
        p_out, f_out, n_out = read_parameters(path)

        cost, row_ind, col_ind = match_latent_variables(p_true, f_true, n_true, p_out, f_out, n_out, norm=1)
        matched_latent_variables.append(np.array(col_ind)) # because row_ind = range(L)

        ll.append(ll_value)

        if epoch > 50:
            break

matches = []
for i in range(len(matched_latent_variables)):
    matches.append(np.sum(matched_latent_variables[-1] == matched_latent_variables[i]))

plt.plot(iters, matches, 'b-', linewidth=2.5)
plt.plot(iters, ll, 'r-', linewidth=2.5)
plt.legend(('correct matches', 'nll'), fontsize='xx-large', loc=1)
# plt.title('16 latent variables', fontsize='xx-large')
plt.xlabel('epochs', fontsize='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks([0.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0], fontsize='xx-large')
plt.ylim((0.0, 25.0))

plt.rcParams.update({'font.size': 18})

plt.show()
