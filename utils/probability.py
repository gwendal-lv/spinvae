"""
Utility functions related to probabilities and statistics, e.g. log likelihoods, ...
"""

import numpy as np

import torch
from torch.distributions.normal import Normal


__log_2_pi = np.log(2*np.pi)


def standard_gaussian_log_probability(samples):
    """
    Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (zero-mean, identity covariance matrix).
    """
    return -0.5 * (samples.shape[1] * __log_2_pi + torch.sum(samples**2, dim=1))


def gaussian_log_probability(samples, mu, log_var):
    """
    Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (diagonal covariance matrix).
    """
    # if samples and mu do not have the same size,
    # torch automatically properly performs the subtract if mu is 1 dim smaller than samples
    return -0.5 * (samples.shape[1] * __log_2_pi +
                   torch.sum( log_var + ((samples - mu)**2 / torch.exp(log_var)), dim=1))


class MMD:
    def __init__(self, kernel='inverse_quadratic', sigma=1.0, unbiased=True, normalize=False):
        """

        :param kernel: 'inverse_multiquadratic' or 'gaussian_RBF'
        :param sigma: Expected standard deviation of each 1d distribution
        :param unbiased: If True, used an unbiased estimator for expectancies, but the estimated MMD can be negative.
        :param normalize: If True, divides the resulting MMD by the length of given random vectors.
        """
        self.sigma = sigma
        self.unbiased = unbiased
        self.normalize = normalize
        if kernel.lower() == 'inverse_quadratic':
            self.kernel_function = self._inverse_quadratic_kernel
        elif kernel.lower() == 'gaussian_rbf':
            self.kernel_function = self._gaussian_rbf
        else:
            raise NotImplementedError("Unavailable kernel '{}'".format(kernel))

    def __call__(self, x_samples, y_samples=None):
        """ Computes the Maximum Mean Discrepancy between the two sets of samples. If y is None, it will
        drawn from a standard normal distribution (identity covariance matrix).

        Expected size for input tensors is N_minibatch x D. """
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
        return mmd if not self.normalize else mmd / D

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
