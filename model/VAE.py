from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import contextlib
from torch.autograd import profiler

from nflows.flows.realnvp import SimpleRealNVP
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import model.base
import model.loss
import model.flows
from utils.probability import gaussian_log_probability, standard_gaussian_log_probability



class BasicVAE(model.base.TrainableModel):
    """ A standard VAE that uses some given encoder and decoder networks.
     The latent probability distribution is modeled as dim_z independent Gaussian distributions. """

    def __init__(self, encoder, dim_z, decoder, normalize_latent_loss,
                 concat_midi_to_z0=False,
                 latent_loss_type: Optional[str] = None,
                 train_config=None):
        super().__init__(train_config=train_config, model_type='ae')
        # No size checks performed. Encoder and decoder must have been properly designed
        self.encoder = encoder
        self.dim_z = dim_z
        self.decoder = decoder
        self.normalize_latent_loss = normalize_latent_loss
        self.is_profiled = False
        self.concat_midi_to_z0 = concat_midi_to_z0

        if latent_loss_type is None:
            self.latent_criterion = None  # must be assigned by child class
        elif latent_loss_type.lower() == 'dkl':
            # TODO try don't normalize (if reconstruction loss is not normalized either)
            self.latent_criterion = model.loss.GaussianDkl(normalize=self.normalize_latent_loss)
        else:
            raise NotImplementedError("Latent loss '{}' unavailable".format(latent_loss_type))

    def _encode_and_sample(self, x, sample_info=None):
        n_minibatch = x.size()[0]
        with profiler.record_function("ENCODING") if self.is_profiled else contextlib.nullcontext():
            # Don't ask for requires_grad or this tensor becomes a leaf variable (it will require grad later)
            z_0_mu_logvar = torch.empty((n_minibatch, 2, self.dim_z), device=x.device, requires_grad=False)
            if not self.concat_midi_to_z0:
                z_0_mu_logvar = self.encoder(x)
            else:  # insert midi notes if required
                z_0_mu_logvar[:, :, 2:] = self.encoder(x)
                if sample_info is None:  # missing MIDI notes are tolerated for graphs and summaries
                    z_0_mu_logvar[:, :, [0, 1]] = 0.0
                else:  # MIDI pitch and velocity models: free-mean and unit-variance scaled in [-1.0, 1.0]
                    # TODO extend this to work with multiple MIDI notes?
                    # Mean is simply scaled to [-1.0, 1.0] (min/max normalization)
                    midi_pitch_and_vel_mu = - 1.0 + 2.0 * sample_info[:, [1, 2]].float() / 127.0
                    z_0_mu_logvar[:, 0, [0, 1]] = midi_pitch_and_vel_mu
                    # log(var) corresponds to a unit standard deviation in the original [0, 127] MIDI domain
                    z_0_mu_logvar[:, 1, [0, 1]] = np.log(4.0 / (127 ** 2))
            # Separate mean and standard deviation
            mu0 = z_0_mu_logvar[:, 0, :]
            sigma0 = torch.exp(z_0_mu_logvar[:, 1, :] / 2.0)
        with profiler.record_function("LATENT_SAMPLING") if self.is_profiled else contextlib.nullcontext():
            if self.training:
                # Sampling from the q_phi(z|x) probability distribution - with re-parametrization trick
                eps = Normal(torch.zeros(n_minibatch, self.dim_z, device=mu0.device),
                             torch.ones(n_minibatch, self.dim_z, device=mu0.device)).sample()
                z_0_sampled = mu0 + sigma0 * eps
            else:  # eval mode: no random sampling
                z_0_sampled = mu0
        return z_0_mu_logvar, z_0_sampled

    def forward(self, x, sample_info=None):
        """ Encodes the given input into a q_phi(z|x) probability distribution,
        samples a latent vector from that distribution, and finally calls the decoder network.

        For compatibility, it returns zK_sampled = z_sampled and the log abs det jacobian(T) = 0.0
        (T = identity)

        :returns: z_mu_logvar, z_sampled, zK_sampled=z_sampled, logabsdetjacT=0.0, x_out (reconstructed spectrogram)
        """
        z_mu_logvar, z_sampled = self._encode_and_sample(x, sample_info)
        with profiler.record_function("DECODING") if self.is_profiled else contextlib.nullcontext():
            x_out = self.decoder(z_sampled)
        return z_mu_logvar, z_sampled, z_sampled, torch.zeros((z_sampled.shape[0], 1), device=x.device), x_out

    def latent_loss(self, z_0_mu_logvar, *args):
        """ *args are not used (they exist for compatibility with flow-based latent spaces). """
        # Default: divergence or discrepancy vs. zero-mean unit-variance multivariate gaussian
        return self.latent_criterion(z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])

    @property
    def is_flow_based_latent_space(self):
        return False


class FlowVAE(BasicVAE):
    """
    A VAE with flow transforms in the latent space.
    q_ZK(z_k) is a complex distribution and does not have a closed-form expression.

    The loss does not rely on a Kullback-Leibler divergence but on a direct log-likelihood computation.
    """

    def __init__(self, encoder, dim_z, decoder, normalize_latent_loss: bool, flow_arch: str,
                 concat_midi_to_z0=False, flows_internal_dropout_p=0.0,
                 train_config=None):
        """

        :param encoder:  CNN-based encoder, output might be smaller than dim_z (if concat MIDI pitch/vel)
        :param dim_z: Latent vectors dimension, including possibly concatenated MIDI pitch and velocity.
        :param decoder:
        :param normalize_latent_loss:
        :param flow_arch: Full string-description of the flow, e.g. 'realnvp_4l200' (flow type, number of flows,
            hidden features count, and batch-norm options '_BNbetween' and '_BNinternal' ...)
        :param concat_midi_to_z0: If True, encoder output mu and log(var) vectors must be smaller than dim_z, for
            this model to append MIDI pitch and velocity (see corresponding mu and log(var) in forward() implementation)
        """
        # Latent loss will be init from this class' ctor
        super().__init__(encoder, dim_z, decoder, normalize_latent_loss, concat_midi_to_z0=concat_midi_to_z0,
                         latent_loss_type=None, train_config=train_config)
        # Latent flow setup
        self.flow_type, self.flow_num_layers, self.flow_num_hidden_units_per_layer, self.flow_bn_between_layers, \
            self.flow_bn_inside_layers, self.flow_output_bn = model.flows.parse_flow_args(flow_arch)
        # TODO finir ça proprement
        if self.flow_type.lower() == 'maf':
            transforms = []
            for _ in range(self.flow_layers_count):
                transforms.append(ReversePermutation(features=self.dim_z))
                # TODO add all options
                transforms.append(MaskedAffineAutoregressiveTransform(
                    features=self.dim_z, hidden_features=self.flow_num_hidden_units_per_layer))
            self.flow_transform = CompositeTransform(transforms)
        elif self.flow_type.lower() == 'realnvp':
            # TODO use a custom RealNVP instance....
            # BN between flow layers prevents reversibility during training
            self.flow_transform = model.flows.CustomRealNVP(
                features=self.dim_z, hidden_features=self.flow_num_hidden_units_per_layer,
                num_layers=self.flow_num_layers, dropout_probability=flows_internal_dropout_p,
                bn_within_layers=self.flow_bn_inside_layers, bn_between_layers=self.flow_bn_between_layers,
                output_bn=self.flow_output_bn)
        else:
            raise NotImplementedError("Unavailable flow '{}'".format(self.flow_type))

    @property
    def is_flow_based_latent_space(self):
        return True

    @property
    def flow_forward_function(self):
        return self.flow_transform.forward

    @property
    def flow_inverse_function(self):
        return self.flow_transform.inverse

    def forward(self, x, sample_info=None):
        """ Encodes the given input into a q_Z0(z_0|x) probability distribution,
        samples a latent vector from that distribution,
        transforms it into q_ZK(z_K|x) using a invertible normalizing flow,
        and finally calls the decoder network using the z_K samples.

        :param x: Single- or Multi-channel spectrogram tensor
        :param sample_info: Required for MIDI pitch end velocity to be appended to the latent vector. On the last dim,
            index 0 should be a preset UID, index 1 a MIDI pitch, index 2 a MIDI velocity.

        :returns: z0_mu_logvar, z0_sampled, zK_sampled, logabsdetjacT, x_out (reconstructed spectrogram)
        """
        n_minibatch = x.size()[0]
        z_0_mu_logvar, z_0_sampled = self._encode_and_sample(x, sample_info)
        with profiler.record_function("LATENT_FLOW") if self.is_profiled else contextlib.nullcontext():
            # Forward flow (fast with nflows MAF implementation - always fast with RealNVP)
            z_K_sampled, log_abs_det_jac = self.flow_transform(z_0_sampled)
        with profiler.record_function("DECODING") if self.is_profiled else contextlib.nullcontext():
            x_out = self.decoder(z_K_sampled)
        return z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out

    def latent_loss(self, z_0_mu_logvar, *args):
        """

        :param z_0_mu_logvar:
        :param args: z_0_sampled, z_K_sampled, log_abs_det_jac (must be provided in that specific order)
        :return:
        """
        z_0_sampled, z_K_sampled, log_abs_det_jac = args
        # log-probability of z_0 is evaluated knowing the gaussian distribution it was sampled from
        log_q_Z0_z0 = gaussian_log_probability(z_0_sampled, z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])
        # log-probability of z_K in the prior p_theta distribution
        # We model this prior as a zero-mean unit-variance multivariate gaussian
        log_p_theta_zK = standard_gaussian_log_probability(z_K_sampled)
        # Returned is the opposite of the ELBO terms
        if not self.normalize_latent_loss:  # Default, which returns actual ELBO terms
            return -(log_p_theta_zK - log_q_Z0_z0 + log_abs_det_jac).mean()  # Mean over batch dimension
        else:  # Mean over batch dimension and latent vector dimension (D)
            return -(log_p_theta_zK - log_q_Z0_z0 + log_abs_det_jac).mean() / z_0_sampled.shape[1]

