import os
import sys

from noisy_or_model import NoisyOrModel
from logistic_regression_model import LogisticRegressionModel
from logistic_regression_model import sample_from_logistic_regression_model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

''' Learns a noisy-OR model using variational learning with a logistic
    regression recognition network, as well as with variance normalization
    and input-dependent signal centering. '''


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
    'p_learning_rate': 0.04,
    'q_rate_factor': 1.0,
    'c_learning_rate': 0.0001,
    'C_alpha': 0.8,
    'C_size': 100,
    'batch_size': 20,
    'save_every': 100,
    'epochs': 1000,
    'heldout': 100,
    'input_dependent_centering': True,
}
batch_size = config['batch_size']

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
samples = np.copy(samples[config['heldout'] :])
np.random.shuffle(samples)
N = samples.shape[0] # number of samples
N_heldout = samples_heldout.shape[0] # number of heldout samples

mean_sample = np.mean(samples, 0)

# Create models.
P = NoisyOrModel(dim_latent, dim_observed)
Q = LogisticRegressionModel(dim_latent, dim_observed)
C = nn.Sequential(
    nn.Linear(dim_observed, config['C_size']),
    nn.Tanh(),
    nn.Linear(config['C_size'], 1)
)

# Define optimizers.
P_optimizer = optim.Adam([P.prior_parameters, P.failure_parameters, P.noise_parameters], lr = config['p_learning_rate'])
Q_optimizer = optim.Adam([Q.weight_parameters, Q.bias_parameters], lr = config['p_learning_rate'] * config['q_rate_factor'])
C_optimizer = optim.SGD(C.parameters(), lr = config['c_learning_rate'])

C_control = 0.0
C_alpha = config['C_alpha']

sys.stdout = open(output_path + '/' + 'log.txt', 'w')
print config

# Train.
for epoch in range(config['epochs']):
    print 'Epoch', epoch

    if (epoch + 1) % config['save_every'] == 0:
        save_model(P, output_path + '/' + 'model_epoch' + str(epoch + 1) + '.dat')

    # Define useful variables.
    shuffle = torch.randperm(N)
    batch = torch.zeros(batch_size, dim_observed)
    centered_batch = torch.zeros(batch_size, dim_observed)
    epoch_nll = 0.0

    # Optimize.
    for i in range(0, N, batch_size):
        # if epoch <= 10 and i % (batch_size * 10) == 0:
        #     save_model(P, output_path + '/' + 'model_epoch' + str(epoch + 1) + '_iter' + str(i + 1) + '.dat')

        for j in range(i, min(i + batch_size, N)):
            batch[j - i] = torch.tensor(samples[shuffle[j]])
            centered_batch[j - i] = torch.tensor(samples[shuffle[j]] - mean_sample)

        P_optimizer.zero_grad()
        Q_optimizer.zero_grad()
        C_optimizer.zero_grad()

        h = sample_from_logistic_regression_model(Q, centered_batch)

        P_nll = P.forward(h, batch)
        Q_nll = Q.forward(h, centered_batch)
        epoch_nll += torch.sum(P_nll).item() - torch.sum(Q_nll).item()

        # Variance reduction techniques.
        Q_reweighting_noncentered = -(P_nll - Q_nll)

        if C_control == 0.0:
          C_control = torch.mean(Q_reweighting_noncentered)
          C_var = torch.var(Q_reweighting_noncentered)
        else:
          C_control = C_alpha * C_control + (1.0 - C_alpha) * torch.mean(Q_reweighting_noncentered)
          C_var = C_alpha * C_var + (1.0 - C_alpha) * torch.var(Q_reweighting_noncentered)

        Q_reweighting_noncentered = Q_reweighting_noncentered - C_control
        if C_var > 1.0:
          Q_reweighting_noncentered = Q_reweighting_noncentered / torch.sqrt(C_var)

        C_out = C.forward(centered_batch).view(batch_size)
        Q_reweighting_centered = Q_reweighting_noncentered - C_out

        # Update models.
        P_sum = torch.sum(P_nll)
        P_sum.backward(torch.tensor([1.0 / batch_size]))

        if config['input_dependent_centering']:
            Q_sum = torch.dot(Q_nll, Q_reweighting_centered.detach())
            Q_sum.backward(torch.tensor([1.0 / batch_size]))
        else:
            Q_sum = torch.dot(Q_nll, Q_reweighting_noncentered.detach())
            Q_sum.backward(torch.tensor([1.0 / batch_size]))

        C_MSE = nn.MSELoss()(C_out, Q_reweighting_noncentered.detach())
        C_MSE.backward()

        # Optimize.
        P_optimizer.step()
        Q_optimizer.step()
        C_optimizer.step()
    epoch_nll /= N

    # Compute heldout likelihood.
    epoch_nll_heldout = 0.0
    if N_heldout != 0:
        batch_heldout = torch.tensor(samples_heldout).float()
        centered_batch_heldout = torch.tensor(samples_heldout - mean_sample).float()
        h_heldout = sample_from_logistic_regression_model(Q, centered_batch_heldout)
        P_nll_heldout = P.forward(h_heldout, batch_heldout)
        Q_nll_heldout = Q.forward(h_heldout, centered_batch_heldout)
        epoch_nll_heldout = torch.sum(P_nll_heldout).item() - torch.sum(Q_nll_heldout).item()
        epoch_nll_heldout /= N_heldout


    print 'Log likelihood (train, heldout)', -epoch_nll, -epoch_nll_heldout
    sys.stdout.flush()

save_model(P, output_path + '/' + 'model_epoch' + str(config['epochs'] + 1) + '.dat')
