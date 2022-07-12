"""
Utility functions and classes related to probabilities and statistics, e.g. log likelihoods, ...
"""

from abc import abstractmethod

import numpy as np

import torch
from torch.distributions.normal import Normal


__log_2_pi = np.log(2*np.pi)


def standard_gaussian_log_probability(samples, add_log_2pi_term=True):
    """ Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (zero-mean, identity covariance matrix). """
    if len(samples.shape) > 2:
        raise NotImplementedError()
    return -0.5 * ((samples.shape[1] * __log_2_pi if add_log_2pi_term else 0.0) +
                   torch.sum(samples**2, dim=1))


def gaussian_log_probability(samples, mu, log_var, add_log_2pi_term=True):
    """ Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (diagonal covariance matrix). """
    # if samples and mu do not have the same size,
    # torch automatically properly performs the subtract if mu is 1 dim smaller than samples
    if len(samples.shape) > 2:
        raise NotImplementedError()
    return -0.5 * ((samples.shape[1] * __log_2_pi if add_log_2pi_term else 0.0) +
                   torch.sum( log_var + ((samples - mu)**2 / torch.exp(log_var)), dim=1))


def gaussian_unitvar_log_probability(samples, mu, add_log_2pi_term=True):
    if len(samples.shape) > 2:
        raise NotImplementedError()
    return -0.5 * ((samples.shape[1] * __log_2_pi if add_log_2pi_term else 0.0) +
                   torch.sum( ((samples - mu)**2), dim=1))


def standard_gaussian_dkl(mu, var, reduction='none'):
    """
    Computes the Dkl between a factorized gaussian distribution (given in input args as 2D tensors) and
    the standard gaussian distribution.

    :param reduction: If 'none', return a batch of KLDs (1 value / batch item). If 'mean', returns a single
        batch-averaged KLD value.
    """
    assert len(mu.shape) == 2 and len(var.shape) == 2, "This method accepts flat (2D) batched tensors only"
    Dkl = 0.5 * torch.sum(var + torch.square(mu) - torch.log(var) - 1.0, dim=1)
    if reduction == 'none':
        return Dkl
    elif reduction == 'mean':
        return torch.mean(Dkl)
    else:
        raise NotImplementedError(reduction)


def standard_gaussian_dkl_2d(mu, var, dim=(1, 2, 3), reduction='none'):
    """
    Computes the KLD of input feature maps vs. a factorized std gaussian distribution of the same shape.

    :param dim: Dimensions that will be used to retrieve distributions' parameters (e.g. H and W of a feature maps
        can provide a proba distribution). Dims that are not included
        indicate independent distributions (e.g. different channels can provide independent distributions).
    :param reduction: see standard_gaussian_dkl(...)
    """
    assert len(mu.shape) == 4 and len(var.shape) == 4, "This method accepts 2D feature maps tensors (4D, w/ batch) only"
    assert 0 not in dim, "Batch dimension should not be included in the KLD computation (independent batch items)."
    Dkl = 0.5 * torch.sum(var + torch.square(mu) - torch.log(var) - 1.0, dim=dim)
    if reduction == 'none':
        return Dkl
    elif reduction == 'mean':
        return torch.mean(Dkl, dim=0)
    else:
        raise NotImplementedError(reduction)

class ProbabilityDistribution:
    def __init__(self, mu_activation=torch.nn.Hardtanh(), reduction='mean'):
        """ Generic class to use decoder outputs as parameters of a probability distribution. """
        self.mu_act = mu_activation
        self.reduction = reduction

    @abstractmethod
    def apply_activations(self, nn_raw_output: torch.Tensor):
        """ Applies the proper activations the returns the distribution parameters. """
        pass

    @abstractmethod
    def NLL(self, x: torch.Tensor, distribution_parameters: torch.Tensor):
        pass

    @abstractmethod
    def sample(self, distribution_parameters: torch.Tensor):
        pass

    @property
    @abstractmethod
    def num_parameters(self):
        """ Number of parameters for each output pixel or time value (number of raw output channels). """
        pass

    def _reduce(self, log_probs: torch.Tensor):
        if self.reduction == 'mean':
            return torch.mean(log_probs)
        elif self.reduction == 'none':
            return log_probs
        else:
            raise NotImplementedError("Unavailable reduction '{}'".format(self.reduction))


class GaussianUnitVariance(ProbabilityDistribution):
    def __init__(self, mu_activation=torch.nn.Hardtanh(), reduction='mean'):
        super().__init__(mu_activation, reduction)

    def apply_activations(self, nn_raw_output: torch.Tensor):
        assert nn_raw_output.shape[1] == 1, "Parameters should be 1-ch (Gaussian means are the only free parameters)"
        return self.mu_act(nn_raw_output)

    def NLL(self, distribution_parameters: torch.Tensor, x: torch.Tensor):
        # We can evaluate the prob in a single operation - don't need to compute it channel-by-channel
        # Distribution parameters are expected to be means only. This computation is very close to being an MSE loss...
        distrib = torch.distributions.Normal(distribution_parameters, torch.ones_like(distribution_parameters))
        return - self._reduce(distrib.log_prob(x))

    def sample(self, distribution_parameters: torch.Tensor):
        return distribution_parameters  # No sampling: parameters contain means only

    @property
    def num_parameters(self):
        return 1



class MMD:
    def __init__(self, kernel='inverse_quadratic', sigma=1.0, unbiased=True):
        """

        :param kernel: 'inverse_multiquadratic' or 'gaussian_RBF'
        :param sigma: Expected standard deviation of each 1d distribution
        :param unbiased: If True, used an unbiased estimator for expectancies, but the estimated MMD can be negative.
        """
        self.sigma = sigma
        self.unbiased = unbiased
        if kernel.lower() == 'inverse_quadratic':
            self.kernel_function = self._inverse_quadratic_kernel
        elif kernel.lower() == 'gaussian_rbf':
            self.kernel_function = self._gaussian_rbf
        else:
            raise NotImplementedError("Unavailable kernel '{}'".format(kernel))

    def __call__(self, x_samples, y_samples=None):
        """ Computes the Maximum Mean Discrepancy between the two sets of samples. If y is None, it will
        be drawn from a standard normal distribution (identity covariance matrix).

        Expected size for input tensors is N_minibatch x D. """
        # FIXME gets stuck around a caller of this (exceptions are raise, where are they catched???)
        N, D = x_samples.shape[0], x_samples.shape[1]  # minibatch size, length of random vector
        if y_samples is None:  # generate y if None
            y_samples = Normal(torch.zeros((N, D), device=x_samples.device),
                               torch.ones((N, D), device=x_samples.device)).sample()
        # Compute all kernel values
        k_x_x = self._compute_kernel_values(x_samples, x_samples)
        k_x_y = self._compute_kernel_values(x_samples, y_samples)
        k_y_y = self._compute_kernel_values(y_samples, y_samples)
        # unbiased expectancy computation - constant 1.0 diagonal values are removed
        if self.unbiased:
            k_x_x, k_y_y = k_x_x.fill_diagonal_(0.0), k_y_y.fill_diagonal_(0.0)
            mmd = k_x_x.sum() / (N * (N-1)) - 2 * k_x_y.sum() / (N * N) + k_y_y.sum() / (N * (N-1))
        else:
            mmd = (k_x_x.sum() - 2 * k_x_y.sum() + k_y_y.sum()) / (N * N)
        return mmd

    def _compute_kernel_values(self, x, y):
        N, D = x.shape[0], x.shape[1]  # minibatch size, length of random vector
        # Repeat x1, x2, ..., xN vectors along dimension 1 (y1, ..., yN along dimension 0)
        # but we don't use tensor.repeat because it copies data.
        # unsqueeze and expand give the same result, but expand "may result in incorrect behavior. If you need
        #    to write to the tensors, please clone them first."
        x_3d = torch.unsqueeze(x, 1).expand(-1, N, -1)
        y_3d = torch.unsqueeze(y, 0).expand(N, -1, -1)
        x_y_squared_distances = torch.sum((x_3d - y_3d) ** 2, dim=2)
        return self.kernel_function(x_y_squared_distances, D)

    def _inverse_quadratic_kernel(self, x_y_squared_distances, D):
        C = 2 * D * self.sigma ** 2  # From Wasserstein Auto-Encoders https://arxiv.org/abs/1711.01558
        return C / (C + x_y_squared_distances)

    def _gaussian_rbf(self, x_y_squared_distances, D):
        C = 2 * D * self.sigma ** 2  # From Wasserstein Auto-Encoders https://arxiv.org/abs/1711.01558
        return torch.exp(- x_y_squared_distances / C)



if __name__ == "__main__":
    _x = torch.Tensor([[0.1, 0.2, 0.3],
                       [1.0, 2.0, 3.0],
                       [1.1, 2.1, 3.1],
                       [-1.2, -2.2, -3.2]])
    _y = torch.Tensor([[-0.1, -0.2, -0.3],
                       [-1.0, -2.0, -3.0],
                       [-1.1, -2.2, -3.3],
                       [-1.5, -2.6, -3.7]])
    _mmd = MMD()
    m = _mmd(_x, None)
    print(m)
