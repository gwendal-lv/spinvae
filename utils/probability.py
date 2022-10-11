"""
Utility functions and classes related to probabilities and statistics, e.g. log likelihoods, ...
"""

from abc import abstractmethod
from typing import Optional, List

import numpy as np

import torch
import warnings
from torch import nn
import torch.nn.functional as F
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


def reduce_dkl_vectors(Dkl, reduction: str):
    if reduction == 'none':
        return Dkl
    elif reduction == 'mean':
        return torch.mean(Dkl)
    else:
        raise NotImplementedError(reduction)


def standard_gaussian_dkl(mu, var, reduction='none'):
    """
    Computes the Dkl between a factorized gaussian distribution (given in input args as 2D tensors) and
    the standard gaussian distribution.

    :param reduction: If 'none', return a batch of KLDs (1 value / batch item). If 'mean', returns a single
        KLD value averaged over all batches.
    """
    assert len(mu.shape) == 2 and len(var.shape) == 2, "This method accepts flat (2D) batched tensors only"
    Dkl = 0.5 * torch.sum(var + torch.square(mu) - torch.log(var) - 1.0, dim=1)
    return reduce_dkl_vectors(Dkl, reduction)


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


def gaussian_dkl(mu1, var1, mu2, var2, reduction='none'):
    """ Returns the KLD between two multivariate gaussian with diagonal covariance matrices, so var1 and var2
     are expected to be vectors (same shape as mu1 and mu2).

     This KLD is the sum of the KLDs of all univariate distributions (simplified formula w/ diag covar matrices)
    :param reduction: If 'none', return a batch of KLDs (1 value / batch item). If 'mean', returns a single
        KLD value averaged over all batches.
     """
    assert len(mu1.shape) == len(var1.shape) == len(mu2.shape) == len(var2.shape) == 2
    Dkl = 0.5 * torch.sum(torch.log(var2) - torch.log(var1) + (var1 + torch.square(mu2 - mu1)) / var2 - 1, dim=1)
    return reduce_dkl_vectors(Dkl, reduction)


def symmetric_gaussian_dkl(mu1, var1, mu2, var2, reduction='none'):
    """ Computes 0.5( KLD(P||Q) + KLD(Q||P) ).
     See gaussian_dkl(...) about input args.  """
    assert len(mu1.shape) == len(var1.shape) == len(mu2.shape) == len(var2.shape) == 2
    mu_squared_diff = torch.square(mu2 - mu1)
    Dkl = 0.25 * torch.sum((var1 + mu_squared_diff) / var2 + (var2 + mu_squared_diff) / var1 - 2, dim=1)
    return reduce_dkl_vectors(Dkl, reduction)



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
    def NLL(self, distribution_parameters: torch.Tensor, x: torch.Tensor, _0: Optional[torch.Tensor] = None):
        pass

    @abstractmethod
    def get_mode(self, distribution_parameters: torch.Tensor, _0: Optional[torch.Tensor] = None):
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


def logistic_density(x: torch.Tensor, mu: torch.Tensor, s: torch.Tensor):
    """ Evaluates the probability of each element of x given the mean and s values (same shape as x) """
    exp_x_norm_center = torch.exp(- (x - mu) * torch.reciprocal(s))  # "Normalized" and centered variable, then exp it
    return exp_x_norm_center * torch.reciprocal(s * torch.square(1.0 + exp_x_norm_center))


class DiscretizedLogisticMixture(ProbabilityDistribution):
    def __init__(self, n_mixtures: int, reduction='mean', prob_mass_leakage=False,
                 min_log_scale=-5.0, min_component_proba=1e-5):
        """

        :param n_mixtures:
        :param prob_mass_leakage: If False, probability mass from "-infty->0.0 and 1.0->+infty" will be aggregated
            to the 0.0 and 1.0 target values (see PixelCNN++ paper). Otherwise, some probability mass will be lost
            but the 0.0 and 1.0 target values won't have an artificially increased probability.
        """
        self.min_log_scale = min_log_scale
        self.min_component_proba = min_component_proba
        self.prob_mass_leakage = prob_mass_leakage
        if self.prob_mass_leakage:
            # No increased probability assigned to 0.0 and 1.0 => use an activation that allows mean values
            # to reach 0.0 and 1.0 easily
            mu_act = nn.Hardtanh(0.0, 1.0)
        else:
            # The means don't need to go to 0.0 and 1.0 exactly for the 0.0 and 1.0 targets to have the highest probs
            mu_act = nn.Sigmoid()
        super().__init__(mu_act, reduction)
        self.n_mix = n_mixtures

    @property
    def num_parameters(self):
        return 3 * self.n_mix

    def _split_mu_s_w(self, distrib_params):
        mu = distrib_params[:, 0:self.n_mix]
        s = distrib_params[:, self.n_mix:2*self.n_mix]
        w = distrib_params[:, self.n_mix*2:3*self.n_mix]
        return mu, s, w

    def apply_activations(self, nn_raw_output: torch.Tensor):
        """ Partial activation: keep log-scales (instead of scales) and does not softmax the mixture weights. """
        distrib_params = torch.empty_like(nn_raw_output, device=nn_raw_output.device)
        mu_raw, s_raw, w_raw = self._split_mu_s_w(nn_raw_output)
        distrib_params[:, 0:self.n_mix] = self.mu_act(mu_raw)
        # Only set minimum scale (pixelCNN++: log-scale minimum is -7.0, corresponds to 1e-4 scale)
        distrib_params[:, self.n_mix:2*self.n_mix] = torch.clamp(s_raw, min=self.min_log_scale)
        # PixelCNN++ applies a tanh activation to the mixture weights (before softmax) - to prevent
        # mixture components from disappearing entirely?
        distrib_params[:, self.n_mix*2:3*self.n_mix] = torch.tanh(w_raw)  # FIXME quite strong constraint....
        return distrib_params

    def _check_input_shapes(self, x_target: torch.Tensor, x_card: torch.Tensor, distrib_params: torch.Tensor):
        assert x_target.shape == x_card.shape
        assert x_target.shape[1] == 1, "A single-channel input is required"
        assert distrib_params.shape[1] == self.num_parameters
        assert x_target.shape[2:] == distrib_params.shape[2:]

    def _expand_x_to_mixture(self, x_target: torch.Tensor, x_card: torch.Tensor):
        """ Expands dim 1 of these tensors to self.n_mix """
        x_shape = list(x_target.shape)
        x_shape[1] = self.n_mix
        return x_target.expand(x_shape), x_card.expand(x_shape)

    def _linear_prob_per_component(self, x_target: torch.Tensor, x_card: torch.Tensor, mu, s_inv):
        assert x_target.shape == x_card.shape == mu.shape == s_inv.shape
        # Pre-computations
        half_delta_x = 0.5 * torch.reciprocal(x_card - 1.0)
        x_centered = x_target - mu
        # Then compute CDF and probability from it
        cdf_right = torch.sigmoid((x_centered + half_delta_x) * s_inv)
        cdf_left = torch.sigmoid((x_centered - half_delta_x) * s_inv)
        if not self.prob_mass_leakage:  # PixelCNN++: probabilities of all possible input targets will sum to 1
            # Masks to handle special "0" and "max value" cases (will be assigned all probability mass up to infty)
            x_target_int = torch.round(x_target * (x_card - 1.0)).int()
            x_is_rightmost = x_target_int >= (x_card - 1)
            # Can't use the binary mask to partially set cdf_right to 1.0 (in-place op would break grad computation)
            #  so we have
            cdf_right = 1.0 - ((1.0 - cdf_right) * (~x_is_rightmost))
            x_is_leftmost = x_target_int <= 0
            cdf_left = cdf_left * (~x_is_leftmost)
        return cdf_right - cdf_left  # p_x_target

    def prob(self, x_target: torch.Tensor, x_card: torch.Tensor, distrib_params: torch.Tensor,
             reduce_mixture=True, params_fully_activated=False):
        """ Return the probabilities of a mixture of logistics.
        For visualisation purposes, not numerically stable

        :param x_target: Shape (N, 1, *). Values must be in [0.0; 1.0] where 1.0 corresponds to the
            int value stored in (x_card - 1) at the same location.
        :param x_card: same shape as x_target
        :param distrib_params: Shape (N, M*3, *)  where M is the number of mixtures. Must have been pre-activated
            using apply_activation (which is not a full activation)
        :param reduce_mixture: Whether to sum the mixture or not before returning the result
        :param params_fully_activated: If True, this function will consider that the given parameters are fully
            activated (e.g. use this when entering manual params, for visualisation)
        :return: Linear probabilities, shape: (N, 1, *) if reduce_mixture else (N, M, *)
        """
        self._check_input_shapes(x_target, x_card, distrib_params)
        # Retrieve target and distrib tensors (all are now the same size, mixture not mixed yet)
        x_target, x_card = self._expand_x_to_mixture(x_target, x_card)
        mu, s, w = self._split_mu_s_w(distrib_params)  # s is log-scale, w is non-softmaxed
        # Compute non-logs probs for all mixtures, and apply mixture weights
        if params_fully_activated:
            s_inv = torch.reciprocal(s)
            w_softmaxed = w
        else:
            s_inv = torch.exp(-s)  # logits are considered log-scale
            w_softmaxed = torch.softmax(w, dim=1)
        p_x_target = self._linear_prob_per_component(x_target, x_card, mu, s_inv)
        p_x_target = p_x_target * w_softmaxed
        # Return separate or summed mixture components
        if reduce_mixture:
            return torch.sum(p_x_target, dim=1, keepdim=True)
        else:
            return p_x_target

    def log_prob(self, x_target: torch.Tensor, x_card: torch.Tensor, distrib_params: torch.Tensor):
        """  input args: see self.prob(...). Returns a (N x 1 x *) tensor"""
        self._check_input_shapes(x_target, x_card, distrib_params)
        x_target, x_card = self._expand_x_to_mixture(x_target, x_card)
        mu, log_s, w = self._split_mu_s_w(distrib_params)  # s is log-scale, w is non-softmaxed
        # Compute per-mixture, non-log probabilities
        s_inv = torch.exp(-log_s)
        p_x_target = self._linear_prob_per_component(x_target, x_card, mu, s_inv)
        # mixture, weights relative to the max weight (will allow to use the probs instead of log probs)
        w_max, _ = torch.max(w, dim=1, keepdim=True)  # discard indices
        w_max_expanded = w_max.expand(w.shape)
        # Sum of "normalized" (not so small, but <= 1.0) weights multiplied by non-log probabilities
        # And the scales are constrained (see apply_activation(...)) such that the non-log probabilities cannot explode
        #    (so: stable logsumexp does seem necessary, such that we don't need to log then exp the probabilities)
        # log_prob = torch.log(torch.sum(torch.exp(w - w_max_expanded) * p_x_target, dim=1, keepdim=True))
        # FIXME DOC logsumexp maybe necessary
        # FIXME log does not help.... TODO try again without the log?
        # FIXME maybe clamp p_x_target to +1e-5 or +1e-4
        p_x_target = torch.clamp(p_x_target, min=self.min_component_proba)
        log_prob = torch.logsumexp(w - w_max_expanded + torch.log(p_x_target), dim=1, keepdim=True)
        return log_prob + w_max - torch.logsumexp(w, dim=1, keepdim=True)

    def NLL(self, distribution_parameters: torch.Tensor, x: torch.Tensor, x_card: Optional[torch.Tensor] = None):
        assert x_card is not None, "x_card (number of possible discrete values for each element of x) must be provided"
        _log_prob = self.log_prob(x, x_card, distribution_parameters)
        return - self._reduce(_log_prob)

    def get_mode(self, distrib_params: torch.Tensor, x_card: Optional[torch.Tensor] = None):
        """ Computes the probability for each possible input value (discrete input set, reasonable computation)
        and returns the value with the highest probability. This method works with sequence inputs only (3D params).

        :param distrib_params: shape N x 3M x L
        :param x_card: shape L (items at the same position in the sequence must have the same set of discrete values)
        """
        assert x_card is not None, "x_card (number of possible discrete values for each element of x) must be provided"
        assert len(distrib_params.shape) == 3,  "This method requires 1d sequence-like input tensors"
        N, L, device = distrib_params.shape[0], distrib_params.shape[2], distrib_params.device
        assert len(x_card.shape) == 1 and x_card.shape[0] == L
        x_card = x_card.long()
        max_card = x_card.max().item()
        # x will be an image-like 4D tensor (all possible values for each parameter)
        x = torch.ones((N, 1, L, max_card), device=device) * -10.0
        # FIXME pre-compute those linspace values?
        for seq_idx in range(L):
            cardinal = x_card[seq_idx].item()
            x[:, :, seq_idx, 0:cardinal] = torch.linspace(0.0, 1.0, cardinal, device=device)  # Broadcast linspace out
        p_x_mask = (x >= 0.0)
        # Unsqueeze distribution parameters to compute the prob for each possible value
        distrib_params = distrib_params.unsqueeze(dim=-1).expand(-1, -1, -1, max_card)
        x_card_expanded = x_card.view(1, 1, L, 1).expand(N, 1, L, max_card)
        p_x = self.prob(x, x_card_expanded, distrib_params) * p_x_mask
        argmax_p_x = torch.argmax(p_x, dim=3, keepdim=False)
        x_modes = argmax_p_x / (x_card - 1.0)  # broadcast 1D x_card over all batches
        return x_modes



class SoftmaxNumerical(ProbabilityDistribution):
    def __init__(self, cardinalities: List[int], dtype: torch.dtype, reduction='mean'):
        """
        Models values of discrete numerical parameters as categories.
        Currently works for entire sequences only (optimized masking) TODO: implement single-token application

        All numerical values are expected to be in [0.0, 1.0]
        (e.g. set of values is {0.0, 0.33, 0.66, 1.0} if cardinality is 3)

        :param cardinalities: The size of each parameter's set of values
        """
        super().__init__(reduction=reduction)
        self.cardinalities = torch.tensor(cardinalities, dtype=torch.long)
        self.n_tokens = self.cardinalities.shape[0]  # Number of elements (num synth param, pixel, ...) being modeled
        # All these probability distributions have "CNN-like" data shapes
        # (channels=distrib.params dimension is last-1, not last)
        self.softmax_mask = torch.zeros(self.num_parameters, self.n_tokens)  # additive mask
        masking_value = torch.finfo(dtype).min
        for token_idx in range(self.n_tokens):
            self.softmax_mask[self.cardinalities[token_idx]:, token_idx] = masking_value
        # unsqueeze mask to make broadcast explicit (over the batch dimension)
        self.softmax_mask = torch.unsqueeze(self.softmax_mask, dim=0)
        # No label smoothing. Could help... but because of the masking method, smaller cardinalities would
        # have a much lower amount of smoothing (unused masked logits would be smoothed also)
        self.CEloss = nn.CrossEntropyLoss(reduction=self.reduction)

    def _check_full_sequence(self, t: torch.Tensor):
        if t.shape[2] != self.n_tokens:
            raise NotImplementedError("Input tensors must be full-sequence ({} tokens expected)".format(self.n_tokens))

    def apply_activations(self, nn_raw_output: torch.Tensor):
        self._check_full_sequence(nn_raw_output)
        return nn_raw_output + self.softmax_mask.to(nn_raw_output.device)

    def NLL(self, distribution_parameters: torch.Tensor, x: torch.Tensor, _0: Optional[torch.Tensor] = None):
        self._check_full_sequence(distribution_parameters)
        self._check_full_sequence(x)
        # Retrieve class indices from x (whose 2nd dimension is 1: single numerical value ; seq dim is 3rd)
        with torch.no_grad():
            target_classes = torch.round(x[:, 0, :] * (self.cardinalities - 1).to(x.device)).long()
        return self.CEloss(distribution_parameters, target_classes)

    def get_mode(self, distribution_parameters: torch.Tensor, _0: Optional[torch.Tensor] = None):
        self._check_full_sequence(distribution_parameters)
        inferred_classes = torch.argmax(distribution_parameters, dim=1)
        # FIXME remove assert when debugged
        assert (inferred_classes - self.cardinalities.to(inferred_classes.device)).max().item() < 0
        return inferred_classes / (self.cardinalities - 1).to(inferred_classes.device)

    @property
    def num_parameters(self):
        return self.cardinalities.max().item()



class GaussianUnitVariance(ProbabilityDistribution):
    def __init__(self, mu_activation=torch.nn.Hardtanh(), reduction='mean'):
        super().__init__(mu_activation, reduction)

    def apply_activations(self, nn_raw_output: torch.Tensor):
        assert nn_raw_output.shape[1] == 1, "Parameters should be 1-ch (Gaussian means are the only free parameters)"
        return self.mu_act(nn_raw_output)

    def NLL(self, distribution_parameters: torch.Tensor, x: torch.Tensor, _0: Optional[torch.Tensor] = None):
        # We can evaluate the prob in a single operation - don't need to compute it channel-by-channel
        # Distribution parameters are expected to be means only. This computation is very close to being an MSE loss...
        distrib = torch.distributions.Normal(distribution_parameters, torch.ones_like(distribution_parameters))
        return - self._reduce(distrib.log_prob(x))

    def get_mode(self, distribution_parameters: torch.Tensor, _0: Optional[torch.Tensor] = None):
        return distribution_parameters  # Means are modes

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

    _card = 4
    _n_mix = 3
    logistic_mixture = DiscretizedLogisticMixture(_n_mix)
    # batch of 2 elements - "channels" dim is unsqueezed
    _x_target = torch.unsqueeze(torch.unsqueeze(torch.linspace(0.0, 1.0, _card), dim=0), dim=0)
    _x_target = torch.cat((_x_target, _x_target), dim=0)
    _x_card = torch.ones_like(_x_target) * _card

    _distrib_params = torch.empty(_x_target.shape[0], logistic_mixture.num_parameters, _x_target.shape[2])
    for batch_idx in range(_x_target.shape[0]):
        mu0, s0, mix_w0, mu1, s1, mix_w1, mu2, s2, mix_w2 = 2 * (np.random.rand(9) - 0.5)
        # We must use smaller scales.... (the NN should learn this)
        s0, s1, s2 = s0 - 2.5, s1 - 2.5, s2 - 2.5
        _distrib_params[batch_idx, 0, :] = mu0
        _distrib_params[batch_idx, 3, :] = s0
        _distrib_params[batch_idx, 6, :] = mix_w0
        _distrib_params[batch_idx, 1, :] = mu1
        _distrib_params[batch_idx, 4, :] = s1
        _distrib_params[batch_idx, 7, :] = mix_w1
        _distrib_params[batch_idx, 2, :] = mu2
        _distrib_params[batch_idx, 5, :] = s2
        _distrib_params[batch_idx, 8, :] = mix_w2
    _distrib_params.requires_grad = True

    # _p_x = logistic_mixture.prob(_x_target, _x_card, _distrib_params)
    _distrib_params = logistic_mixture.apply_activations(_distrib_params)
    _p_x = logistic_mixture.prob(_x_target, _x_card, _distrib_params, reduce_mixture=False)
    _logp_x = logistic_mixture.log_prob(_x_target, _x_card, _distrib_params)
    _nll = logistic_mixture.NLL(_x_target, _distrib_params, _x_card)
    _logp_x

    # card must be 1D now
    _x_card_1d = torch.Tensor([5, 6, 7, 8])
    _smp = logistic_mixture.get_mode(_distrib_params, _x_card_1d)
    a = 0
