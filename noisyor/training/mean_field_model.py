import torch
import torch.nn as nn

class MeanFieldModel(nn.Module):
    def __init__(self, dim_latent):
        super(MeanFieldModel, self).__init__()
        self.dim_latent = dim_latent
        self.parameters = torch.zeros(self.dim_latent)
        self.parameters.uniform_(0.2, 0.8)

    # Output is the negative log-likelihood of the latent variables.
    def forward(self, latent):
        batch_size = latent.shape[0]
        nll = torch.sum(nn.BCELoss(reduce=False)(self.parameters.repeat(batch_size, 1), latent), 1)
        return nll

def sample_from_mean_field_model(Q, batch_size):
    random_sample = torch.rand((batch_size, Q.dim_latent))
    return torch.lt(random_sample, Q.parameters.repeat(batch_size, 1)).float()
