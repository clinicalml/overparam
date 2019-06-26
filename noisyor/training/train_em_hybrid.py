import os
import sys

from noisy_or_model import NoisyOrModel
from mean_field_model import MeanFieldModel
from mean_field_model import sample_from_mean_field_model

import numpy as np
import scipy.misc
import scipy.special
import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.set_num_threads(1)

''' Learns a noisy-OR model using the E-step of the algorithm described in
    Singliar & Hauskrecht (2006), but using gradient steps for the M-step.
    (See http://www.jmlr.org/papers/volume7/singliar06a/singliar06a.pdf) '''


# Save model parameters.
def save_model(model, save_path):
    model_dim_latent = list(model.prior_parameters.size())[0]
    model_dim_observed = list(model.noise_parameters.size())[0]

    outfile = open(save_path, 'w')
    outfile.write(str(model_dim_latent) + ' ' + str(model_dim_observed) + '\n')

    for i in range(model_dim_latent):
        outfile.write(str(torch.sigmoid(model.prior_parameters[i]).item()) + ' ')
    outfile.write('\n')
    for i in range(model_dim_latent):
        for j in range(model_dim_observed):
            outfile.write(str(torch.sigmoid(model.failure_parameters[i, j]).item()) + ' ')
        outfile.write('\n')
    for i in range(model_dim_observed):
        outfile.write(str(torch.sigmoid(model.noise_parameters[i]).item()) + ' ')
    outfile.write('\n')
    outfile.close()


# Configuration of parameters.
config = {
    'seed': None,
    'learning_rate': 10.0,
    'save_every': 10,
    'epochs': 100,
    'heldout': 1000,
}

# Get arguments.
samples_path = sys.argv[1]
output_path = sys.argv[2]
dim_latent = int(sys.argv[3])
if len(sys.argv) > 4:
    config['seed'] = int(sys.argv[4])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

# Read samples.
samples = np.loadtxt(samples_path, dtype=float)
dim_observed = samples.shape[1]

samples_heldout = np.copy(samples[: config['heldout']])
samples = np.copy(samples[config['heldout'] : ])
np.random.shuffle(samples)
N = samples.shape[0] # number of samples
N_heldout = samples_heldout.shape[0] # number of heldout samples

# Create models.
P = NoisyOrModel(dim_latent, dim_observed)

learning_rate = config['learning_rate']

sys.stdout = open(output_path + '/' + 'log.txt', 'w')
print config

# Train.
for epoch in range(config['epochs']):
    print 'Epoch', epoch
    sys.stdout.flush()

    start = time.time()

    # Save model parameters.
    if (epoch + 1) % config['save_every'] == 0:
        save_model(P, output_path + '/' + 'model_epoch' + str(epoch + 1) + '.dat')

    # Define useful variables.
    epoch_nll = 0.0

    prior_parameters = scipy.special.expit(P.prior_parameters.detach().numpy())
    failure_parameters = P.failure_parameters.detach().numpy()
    noise_parameters = P.noise_parameters.detach().numpy()
    t_failure_parameters = -np.log(scipy.special.expit(failure_parameters))
    t_noise_parameters = -np.log(scipy.special.expit(noise_parameters))

    q_parameters = np.full((N, dim_latent, dim_observed), 1.0 / dim_latent)
    q_parameters = np.divide(q_parameters, np.sum(q_parameters, axis=1, keepdims=True))

    def get_expected_s():
        result = np.zeros((N, dim_latent))
        for i in range(0, dim_latent):
            prob_zero = np.full(N, 1.0 - prior_parameters[i])
            prob_one = np.full(N, prior_parameters[i])
            for j in range(0, dim_observed):
                log_tT = scipy.misc.logsumexp(a=[0.0, -t_noise_parameters[j]], b=[1.0, -1.0])
                tA_div = np.divide(t_failure_parameters[i, j], q_parameters[:, i, j], out=np.full(N, 0.0), where=q_parameters[:, i, j]!=0.0)
                log_tA = scipy.misc.logsumexp(a=np.stack([np.full(N, 0.0), -t_noise_parameters[j] - tA_div]), axis=0, b=np.stack([np.full(N, 1.0), np.full(N, -1.0)]))

                prob_zero = np.multiply(prob_zero, np.exp(np.multiply(samples[:, j], np.multiply(q_parameters[:, i, j], log_tT))))

                prob_one = np.multiply(prob_one, np.exp(np.multiply(samples[:, j], np.multiply(q_parameters[:, i, j], log_tA))))
                prob_one = np.multiply(prob_one, np.exp(np.multiply(1.0 - samples[:, j], -t_failure_parameters[i, j])))

            result[:, i] = np.divide(prob_one, prob_zero + prob_one)
        return result

    # Optimize Q.
    expected_s = get_expected_s()
    current_discrepancy = 1.0
    while current_discrepancy > 1e-2:
        new_q_parameters = np.zeros((N, dim_latent, dim_observed))
        for i in range(0, dim_latent):
            for j in range(0, dim_observed):
                log_tT = scipy.misc.logsumexp(a=[0.0, -t_noise_parameters[j]], b=[1.0, -1.0])
                tA_div = np.divide(t_failure_parameters[i, j], q_parameters[:, i, j], out=np.full(N, 0.0), where=q_parameters[:, i, j]!=0.0)
                log_tA = scipy.misc.logsumexp(a=np.stack([np.full(N, 0.0), -t_noise_parameters[j] - tA_div]), axis=0, b=np.stack([np.full(N, 1.0), np.full(N, -1.0)]))
                tA = np.exp(-t_noise_parameters[j] - tA_div)

                new_q_parameters[:, i, j] = np.multiply(np.multiply(expected_s[:, i], q_parameters[:, i, j]), log_tA - np.multiply(tA_div, np.divide(tA, 1.0 - tA)) - log_tT)
        new_q_parameters = np.divide(new_q_parameters, np.sum(new_q_parameters, axis=1, keepdims=True))
        new_q_parameters = np.multiply(new_q_parameters, np.greater(new_q_parameters, 1e-3))

        current_discrepancy = np.sum(np.square(q_parameters - new_q_parameters))

        q_parameters = np.copy(new_q_parameters)
        expected_s = get_expected_s()

    # Optimize P.
    # Update failure and noise parameters.
    def partial_failure(i, j): # partial F / partial failure_parameter[i, j]
        tA_div = np.divide(t_failure_parameters[i, j], q_parameters[:, i, j], out=np.full(N, 0.0), where=q_parameters[:, i, j]!=0.0)
        tA = np.multiply(np.greater(q_parameters[:, i, j], 1e-3), np.exp(-t_noise_parameters[j] - tA_div))
        return np.dot(expected_s[:, i], -1.0 + np.divide(samples[:, j], 1.0 - tA)) / N

    def partial_noise(j):
        sum = - N + np.sum(samples[:, j])
        for i in range(dim_latent):
            tA_div = np.divide(t_failure_parameters[i, j], q_parameters[:, i, j], out=np.full(N, 0.0), where=q_parameters[:, i, j]!=0.0)
            tA = np.multiply(np.greater(q_parameters[:, i, j], 1e-3), np.exp(-t_noise_parameters[j] - tA_div))
            sum += np.dot(np.multiply(expected_s[:, i], np.multiply(q_parameters[:, i, j], samples[:, j])), np.divide(tA, 1.0 - tA) - np.exp(-t_noise_parameters[j]) / (1.0 - np.exp(-t_noise_parameters[j])))
            sum += np.exp(-t_noise_parameters[j]) / (1.0 - np.exp(-t_noise_parameters[j])) * np.dot(q_parameters[:, i, j], samples[:, j])
        return sum / N

    # Take a single update step w.r.t. each parameter.

    new_failure_parameters = np.copy(failure_parameters)
    new_noise_parameters = np.copy(noise_parameters)

    for i in range(0, dim_latent):
        for j in range(0, dim_observed):
            new_failure_parameters[i, j] += learning_rate * partial_failure(i, j) * (-1.0 / (np.exp(failure_parameters[i, j]) + 1.0))
            new_failure_parameters[i, j] = min(new_failure_parameters[i, j], 10.0)
            new_failure_parameters[i, j] = max(new_failure_parameters[i, j], -10.0)
    for j in range(0, dim_observed):
        new_noise_parameters[j] += learning_rate * partial_noise(j) * (-1.0 / (np.exp(noise_parameters[j]) + 1.0))
        new_noise_parameters[j] = min(new_noise_parameters[j], 10.0)
        new_noise_parameters[j] = max(new_noise_parameters[j], -10.0)

    failure_parameters = np.copy(new_failure_parameters)
    noise_parameters = np.copy(new_noise_parameters)

    # Update prior parameters.
    for i in range(dim_latent):
        prior_parameters[i] = np.sum(expected_s[:, i]) / N

    # Set parameters.
    with torch.no_grad():
        for i in range(dim_latent):
            P.prior_parameters[i] = float(prior_parameters[i])
        for i in range(dim_latent):
            for j in range(dim_observed):
                P.failure_parameters[i, j] = float(failure_parameters[i, j])
        for j in range(dim_observed):
            P.noise_parameters[j] = float(noise_parameters[j])

save_model(P, output_path + '/' + 'model_epoch' + str(config['epochs'] + 1) + '.dat')
