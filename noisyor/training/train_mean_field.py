import os
import sys

from noisy_or_model import NoisyOrModel
from mean_field_model import MeanFieldModel
from mean_field_model import sample_from_mean_field_model

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

''' Learns a noisy-OR model using variational learning with a mean-field
    variational posterior (the latent variables are modeled as independent
    Bernoulli). '''


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
    'learning_rate': 0.001,
    'save_every': 1,
    'C_alpha': 0.8,
    'C_size': 100,
    'epochs': 15,
    'heldout': 1000,
    'q_iterations': 5,
    'q_samples': 20, # samples used in expectation
    'p_samples': 20
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

# Define optimizers.
P_optimizer = optim.Adam([P.prior_parameters, P.failure_parameters, P.noise_parameters], lr = config['learning_rate'])

sys.stdout = open(output_path + '/' + 'log.txt', 'w')
print config

# Train.
for epoch in range(config['epochs']):
    print 'Epoch', epoch

    # Save model parameters.
    if (epoch + 1) % config['save_every'] == 0:
        save_model(P, output_path + '/' + 'model_epoch' + str(epoch + 1) + '.dat')

    # Define useful variables.
    shuffle = torch.randperm(N)
    epoch_nll = 0.0

    for i in range(0, N):
        # Optimize Q.
        Q = MeanFieldModel(dim_latent)
        for q_iter in range(config['q_iterations']):
            shuffle_q = torch.randperm(dim_latent)
            for coord in range(dim_latent): # coordinate being updated
                # Update according to q_k(z_k) ~= exp(E[log p(z, x)]).
                expectation = [0.0, 0.0]
                h = sample_from_mean_field_model(Q, config['q_samples'])

                h[:, shuffle_q[coord]] = 0.0
                P_nll = P.forward(h, torch.tensor(samples[shuffle[i]]).float().repeat(config['q_samples'], 1))
                expectation[0] = -torch.sum(P_nll).item() / config['q_samples']

                h[:, shuffle_q[coord]] = 1.0
                P_nll = P.forward(h, torch.tensor(samples[shuffle[i]]).float().repeat(config['q_samples'], 1))
                expectation[1] = -torch.sum(P_nll).item() / config['q_samples']

                expectation = scipy.special.softmax(expectation)
                Q.parameters[shuffle_q[coord]] = expectation[1]

        # Optimize P.
        P_optimizer.zero_grad()

        h = sample_from_mean_field_model(Q, config['p_samples'])
        P_nll = P.forward(h, torch.tensor(samples[shuffle[i]]).float().repeat(config['p_samples'], 1))
        Q_nll = Q.forward(h)
        epoch_nll += (torch.sum(P_nll).item() - torch.sum(Q_nll).item()) / config['p_samples']

        # Update model.
        P_sum = torch.sum(P_nll)
        P_sum.backward()

        # Optimize.
        P_optimizer.step()
    epoch_nll /= N

    # Compute heldout likelihood.
    epoch_nll_heldout = 0.0
    if N_heldout != 0:
        for i in range(0, N_heldout):
            # Optimize Q.
            Q = MeanFieldModel(dim_latent)
            for q_iter in range(config['q_iterations']):
                shuffle_q = torch.randperm(dim_latent)
                for coord in range(dim_latent): # coordinate being updated
                    # Update according to q_k(z_k) ~= exp(E[log p(z, x)]).
                    expectation = [0.0, 0.0]
                    h = sample_from_mean_field_model(Q, config['q_samples'])

                    h[:, shuffle_q[coord]] = 0
                    P_nll = P.forward(h, torch.tensor(samples_heldout[i]).float().repeat(config['q_samples'], 1))
                    expectation[0] = -torch.sum(P_nll).item() / config['q_samples']

                    h[:, shuffle_q[coord]] = 1
                    P_nll = P.forward(h, torch.tensor(samples_heldout[i]).float().repeat(config['q_samples'], 1))
                    expectation[1] = -torch.sum(P_nll).item() / config['q_samples']

                    expectation = scipy.special.softmax(expectation)
                    Q.parameters[shuffle_q[coord]] = expectation[1]

            # Calculate likelihood.
            h = sample_from_mean_field_model(Q, config['p_samples'])
            P_nll = P.forward(h, torch.tensor(samples_heldout[i]).float().repeat(config['p_samples'], 1))
            Q_nll = Q.forward(h)
            epoch_nll_heldout += (torch.sum(P_nll).item() - torch.sum(Q_nll).item()) / config['p_samples']
        epoch_nll_heldout /= N_heldout


    print 'Log likelihood (train, heldout)', -epoch_nll, -epoch_nll_heldout
    sys.stdout.flush()

save_model(P, output_path + '/' + 'model_epoch' + str(config['epochs'] + 1) + '.dat')
