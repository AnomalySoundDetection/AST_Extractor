
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

        log_q += self.q0.log_prob(z)

        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        """
        Estimates reverse KL divergence, see arXiv 1912.02762
        :param num_samples: Number of samples to draw from base distribution
        :param beta: Annealing parameter, see arXiv 1505.05770
        :param score_fn: Flag whether to include score function in gradient, see
        arXiv 1703.09194
        :return: Estimate of the reverse KL divergence averaged over latent samples
        """

        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_

        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det

        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)

            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det

            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)

        log_p = self.p.log_prob(z)

        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        """
        Alpha divergence when sampling from q
        :param num_samples: Number of samples to draw
        :param dreg: Flag whether to use Double Reparametrized Gradient estimator,
        see arXiv 1810.04152
        :return: Alpha divergence
        """

        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det

        log_p = self.p.log_prob(z)

        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)

            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det

            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)

            w = torch.exp(log_p - log_q)
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))

        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)

        return loss

    def sample(self, num_samples=1):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :return: Samples, log probability
        """

        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det

        return z, log_q

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

        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return z, -log_q


def BuildFlow(latent_size, num_layers):
    base = nf.distributions.base.DiagGaussian(latent_size)

    flows = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([int(latent_size/2), latent_size+512, latent_size+512, latent_size], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(latent_size, mode='swap'))

    flow_model = NormalizingFlow(base, flows)

    return flow_model