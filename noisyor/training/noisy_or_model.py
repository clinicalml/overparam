import torch
import torch.nn as nn

class NoisyOrModel(nn.Module):
    def __init__(self, dim_latent, dim_observed):
        super(NoisyOrModel, self).__init__()
        self.dim_latent = dim_latent
        self.dim_observed = dim_observed

        self.prior_parameters = torch.zeros(self.dim_latent, requires_grad = True)
        self.failure_parameters = torch.zeros((self.dim_latent, self.dim_observed), requires_grad = True)
        self.noise_parameters = torch.zeros(self.dim_observed, requires_grad = True)
        with torch.no_grad():
            self.prior_parameters.uniform_(2.0, 4.0)
            self.failure_parameters.uniform_(2.0, 4.0)
            self.noise_parameters.uniform_(2.0, 4.0)

    # Output is the negative log-likelihood of the data under the model.
    # i.e. -log P(h, x)
    def forward(self, latent, observed):
        batch_size = latent.shape[0]

        batch_prior_parameters = torch.sigmoid(self.prior_parameters).repeat(batch_size, 1)
        prior = torch.sum(nn.BCELoss(reduce=False)(batch_prior_parameters, latent), 1)

        observed_probability_zero = torch.matmul(latent, torch.log(torch.sigmoid(self.failure_parameters)))
        observed_probability_zero = observed_probability_zero + torch.log(torch.sigmoid(self.noise_parameters)).repeat(batch_size, 1)
        observed_probability_one = torch.full((batch_size, self.dim_observed), 1.0) - torch.exp(observed_probability_zero)
        conditional = torch.sum(nn.BCELoss(reduce=False)(observed_probability_one, observed), 1)

        return prior + conditional
