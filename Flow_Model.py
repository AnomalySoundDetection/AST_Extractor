
import torch
import torch.nn as nn
import numpy as np

# from normflows import distributions
import normflows as nf
from normflows import utils


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0, flows, p=None):
        """
        Constructor
        :param q0: Base distribution
        :param flows: List of flows
        :param p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    # Originally: forward_kld
    def forward(self, x):
        """
        Estimates forward KL divergence, see arXiv 1912.02762
        :param x: Batch sampled from target distribution
        :return: Estimate of forward KL divergence averaged over batch
        """

        log_q = torch.zeros(len(x), device=x.device)
        z = x

        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det

        #z = torch.reshape(z, (len(x), -1))
        #print(z.shape)

        #log_prob = self.q0.log_prob(z)
        log_q += self.q0.log_prob(z) 

        loss = -torch.mean(log_q)
        loss = loss.unsqueeze(0)

        #log_prob = torch.mean(log_prob)
        #log_prob = log_prob.unsqueeze(0)

        return loss

    def log_prob(self, x):
        """
        Get log probability for batch
        :param x: Batch
        :return: log probability
        """

        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x

        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)

        return log_q

    def save(self, path):
        """
        Save state dict of model
        :param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model from state dict
        :param path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

    def x_to_z(self, x):

        #log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            #log_q += log_det

        #z = z.squeeze()
        z = torch.reshape(z, (len(x), -1))
        #log_q += self.q0.log_prob(z)  

        return z


def BuildFlow(latent_size, channel, num_layers):
    base = nf.distributions.base.DiagGaussian(latent_size)

    flows = []
    for i in range(num_layers):
        param_map = nf.nets.ConvResidualNet(in_channels=channel//2, hidden_channels=channel//2, out_channels=channel, num_blocks=2)
        #param_map = nf.nets.MLP([int(latent_size/2), 1024, 1024, latent_size], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(channel, mode='swap'))

    flow_model = NormalizingFlow(base, flows)

    return flow_model