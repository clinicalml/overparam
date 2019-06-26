import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, dim_latent, dim_observed):
        super(LogisticRegressionModel, self).__init__()
        self.dim_latent = dim_latent
        self.dim_observed = dim_observed

        self.weight_parameters = torch.zeros((self.dim_observed, self.dim_latent), requires_grad = True)
        self.bias_parameters = torch.zeros(self.dim_latent, requires_grad = True)
        with torch.no_grad():
            self.weight_parameters.uniform_(-0.1, 0.1)
            self.bias_parameters.uniform_(-0.1, 0.1)

    # Output is the conditional negative log-likelihood of the latent variables, given the observed variables.
    # i.e. -log Q(h | x)
    def forward(self, latent, observed):
        batch_size = latent.shape[0]
        hidden_probability_one = torch.sigmoid(torch.matmul(observed, self.weight_parameters) + self.bias_parameters.repeat(batch_size, 1))
        conditional = torch.sum(nn.BCELoss(reduce=False)(hidden_probability_one, latent), 1)
        return conditional

def sample_from_logistic_regression_model(Q, observed):
    batch_size = observed.shape[0]
    hidden_probability_one = torch.sigmoid(torch.matmul(observed, Q.weight_parameters) + Q.bias_parameters.repeat(batch_size, 1))
    random_sample = torch.rand((batch_size, Q.dim_latent))
    return torch.lt(random_sample, hidden_probability_one).float()
