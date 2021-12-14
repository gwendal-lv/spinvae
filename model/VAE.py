from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from nflows.flows.realnvp import SimpleRealNVP
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import model.base
import model.loss
import model.flows
from utils.probability import gaussian_log_probability, standard_gaussian_log_probability, MMD



class BasicVAE(model.base.TrainableModel):
    """
    A standard VAE that uses some given encoder and decoder networks.
     The latent probability distribution is modeled as dim_z independent Gaussian distributions.

    A style vector w is always computed from z0_sampled using an 8-layer MLP (StyleGAN).
     """

    def __init__(self, encoder, dim_z, decoder, style_arch: str,
                 concat_midi_to_z0=False,  # FIXME remove that arg - midi note should be concat to style vectors w
                 train_config=None):
        super().__init__(train_config=train_config, model_type='ae')
        # No size checks performed. Encoder and decoder must have been properly designed
        self.encoder = encoder
        self.dim_z = dim_z
        self.decoder = decoder
        self.concat_midi_to_z0 = concat_midi_to_z0
        if concat_midi_to_z0:
            raise AssertionError("MIDI note should not be concatenated to z0, but to style vector w.")

        # if train_config is None: all losses have to be unavailable
        if train_config is None:
            self.is_encoder_deterministic = True
            self.normalize_losses = None
            self._reconstruction_criterion = None
            self.latent_loss_type = None
            self.latent_loss_compensation_factor = None
            self.latent_criterion = None  # might be assigned by child class
            self.mmd_num_estimates = None
            self.mmd_criterion = None
        else:
            self.is_encoder_deterministic = False  # MMD might reset this to True
            self.latent_loss_compensation_factor = 1.0  # Default value, may be written below
            self.normalize_losses = train_config.normalize_losses
            self.mmd_num_estimates = train_config.mmd_num_estimates
            if not isinstance(self.mmd_num_estimates, int) or self.mmd_num_estimates < 1:
                raise ValueError("train.config.mmd_num_estimates must be a > 1 integer value.")
            # reconstruction criterion
            self._monitoring_recons_criterion = nn.MSELoss(reduction='mean')  # to compare models w/ different losses
            if train_config.reconstruction_loss.lower() == "mse":
                if self.normalize_losses:
                    self._reconstruction_criterion = nn.MSELoss(reduction='mean')
                else:
                    self._reconstruction_criterion = model.loss.L2Loss()
            elif train_config.reconstruction_loss.lower() == "weightedmse":
                raise NotImplementedError()
            else:
                raise ValueError("Reconstruction loss '{}' is not available".format(train_config.reconstruction_loss))
            # Latent criterion
            self.latent_loss_type = train_config.latent_loss
            if self.latent_loss_type.lower() == 'dkl':
                self.latent_criterion = model.loss.GaussianDkl(normalize=self.normalize_losses)
            elif self.latent_loss_type[0:3].lower() == 'mmd':
                if self.latent_loss_type.lower() == 'mmd_determ_enc':
                    self.is_encoder_deterministic = True
                elif self.latent_loss_type.lower() != 'mmd':
                    raise ValueError("Invalid latent loss '{}'".format(self.latent_loss_type))
                self.latent_loss_type = 'mmd'
                self.latent_loss_compensation_factor = train_config.mmd_compensation_factor
                # No need to assign self.latent_criterion (proper function will be called directly)
            else:  # Might be assigned by child class
                self.latent_criterion = None
            self.mmd_criterion = MMD()

        if not self.normalize_losses:
            raise AssertionError("MMD criterion (used for monitoring all VAEs) cannot be un-normalized.")

        # TODO parse style arch
        style_args = style_arch.split('_')
        if style_args[0] != 'mlp':
            raise AssertionError("Style network must be 'mlp'")
        layers_args = style_args[1].split('l')
        style_n_layers = int(layers_args[0])
        style_n_units = int(layers_args[1])
        output_bn = False
        if len(style_args) >= 3:
            if style_args[2].lower() == 'outputbn':
                output_bn = True
            else:
                raise AssertionError("Unrecognized style network argument '{}'".format(style_args[2]))
        self.style_mlp = nn.Sequential()
        for i in range(style_n_layers):
            self.style_mlp.add_module('fc{}'.format(i),
                                      nn.Linear(style_n_units if i > 0 else self.dim_z, style_n_units))
            self.style_mlp.add_module('act{}'.format(i), nn.ReLU())
            if i < (style_n_layers-1) or (i == (style_n_layers-1) and output_bn):
                self.style_mlp.add_module('bn{}'.format(i), nn.BatchNorm1d(style_n_units))

    def _encode_and_sample(self, x, sample_info=None):
        n_minibatch = x.size()[0]
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
        # Sampling in training mode only, and when using a stochastic encoder
        if self.training and not self.is_encoder_deterministic:
            # Sampling from the q_phi(z|x) probability distribution - with re-parametrization trick
            eps = Normal(torch.zeros(n_minibatch, self.dim_z, device=mu0.device),
                         torch.ones(n_minibatch, self.dim_z, device=mu0.device)).sample()
            z_0_sampled = mu0 + sigma0 * eps
        else:  # eval mode: no random sampling
            z_0_sampled = mu0
        return z_0_mu_logvar, z_0_sampled

    def _decode_latent_vector(self, z_sampled):
        w_style = self.style_mlp(z_sampled)
        return w_style, self.decoder(z_sampled, w_style)

    def generate_from_latent_vector(self, z):
        return self._decode_latent_vector(z)[1]  # Don't return style

    def forward(self, x, sample_info=None):
        """ Encodes the given input into a q_phi(z|x) probability distribution,
        samples a latent vector from that distribution, and finally calls the decoder network.

        For compatibility, it returns zK_sampled = z_sampled and the log abs det jacobian(T) = 0.0
        (T = identity)

        :returns: z_mu_logvar, z_sampled, zK_sampled=z_sampled, logabsdetjacT=0.0, x_out (reconstructed spectrogram)
        """
        z_mu_logvar, z_sampled = self._encode_and_sample(x, sample_info)
        w_style, x_out = self._decode_latent_vector(z_sampled)
        # TODO also get and return the style vector? for downstream tasks
        return z_mu_logvar, z_sampled, z_sampled, torch.zeros((z_sampled.shape[0], 1), device=x.device), x_out

    def latent_loss(self, z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac):
        """ Some args might be useless - they exist for compatibility with flow-based latent spaces. """
        # Default: divergence or discrepancy vs. zero-mean unit-variance multivariate gaussian
        if self.latent_loss_type.lower() == 'dkl':
            loss = self.latent_criterion(z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])
        elif self.latent_loss_type.lower() == 'mmd':
            loss = self.mmd(z_K_sampled)
        else:
            raise AssertionError("Cannot compute loss - this class was probably instantiated with a train config.")
        return loss * self.latent_loss_compensation_factor

    def mmd(self, z_samples):
        """ Returns the estimated Maximum Mean Discrepancy between the given samples, and samples drawn for a
        multivariate standard normal distribution. Multiple MMDs may be computed and averaged (see train.config). """
        mmds = [self.mmd_criterion(z_samples) for _ in range(self.mmd_num_estimates)]
        return sum(mmds) / len(mmds)

    def monitoring_reconstruction_loss(self, x_out, x_in):
        """ Returns the usual normalized MSE loss (to compare models with different backprop criteria). """
        return self._monitoring_recons_criterion(x_out, x_in)

    def reconstruction_loss(self, x_out, x_in):
        return self._reconstruction_criterion(x_out, x_in)

    @property
    def is_flow_based_latent_space(self):
        return False

    def additional_latent_regularization_loss(self, z_0_mu_logvar):
        """ Returns an optional additional regularization loss, as configured using ctor args. """
        return torch.zeros((1, ), device=z_0_mu_logvar.device)


class FlowVAE(BasicVAE):
    """
    A VAE with flow transforms in the latent space.
    q_ZK(z_k) is a complex distribution and does not have a closed-form expression.

    The loss does not rely on a Kullback-Leibler divergence but on a direct log-likelihood computation.
    TODO also allow MMD loss
    """

    def __init__(self, encoder, dim_z, decoder,  style_arch: str, flow_arch: str,
                 concat_midi_to_z0=False, train_config=None):
        """
        :param flow_arch: Full string-description of the flow, e.g. 'realnvp_4l200' (flow type, number of flows,
            hidden features count, and batch-norm options '_BNbetween' and '_BNinternal' ...)
        :param concat_midi_to_z0: If True, encoder output mu and log(var) vectors must be smaller than dim_z, for
            this model to append MIDI pitch and velocity (see corresponding mu and log(var) in forward() implementation)
        """
        # Latent loss will be init from this class' ctor
        super().__init__(encoder, dim_z, decoder, style_arch,
                         concat_midi_to_z0=concat_midi_to_z0, train_config=train_config)
        if train_config is None:  # This model won't be usable for training anyway (no loss computation)
            self.flows_internal_dropout_p = 0.0
        else:
            self.flows_internal_dropout_p = train_config.fc_dropout

        # Latent flow setup
        self.flow_type, self.flow_num_layers, self.flow_num_hidden_units_per_layer, self.flow_bn_between_layers, \
            self.flow_bn_inside_layers, self.flow_output_bn = model.flows.parse_flow_args(flow_arch)
        if self.flow_type.lower() == 'maf':  # TODO finir ça proprement
            transforms = []
            for _ in range(self.flow_layers_count):
                transforms.append(ReversePermutation(features=self.dim_z))
                # TODO add all options
                transforms.append(MaskedAffineAutoregressiveTransform(
                    features=self.dim_z, hidden_features=self.flow_num_hidden_units_per_layer))
            self.flow_transform = CompositeTransform(transforms)
        elif self.flow_type.lower() == 'realnvp':
            # BN between flow layers prevents reversibility during training
            self.flow_transform = model.flows.CustomRealNVP(
                features=self.dim_z, hidden_features=self.flow_num_hidden_units_per_layer,
                num_layers=self.flow_num_layers, dropout_probability=self.flows_internal_dropout_p,
                bn_within_layers=self.flow_bn_inside_layers, bn_between_layers=self.flow_bn_between_layers,
                output_bn=self.flow_output_bn)
        else:
            raise NotImplementedError("Unavailable flow '{}'".format(self.flow_type))

        # Special (regularization) losses for flows
        self.latent_flow_input_reg_criterion = None  # default values
        self.latent_flow_input_regul_weight = 0.0
        if train_config is not None:
            if train_config.latent_flow_input_regularization.lower() == 'dkl':
                self.latent_flow_input_reg_criterion = model.loss.GaussianDkl(normalize=self.normalize_losses)
                self.latent_flow_input_regul_weight = train_config.latent_flow_input_regul_weight
            else:  # Can be 'BN' (placed near encoder output) or 'None' - or train_config was not given at all
                pass

            # latent_loss already assigned by BasicVAE mother class
            if self.latent_loss_type.lower() == 'dkl':
                raise AssertionError("Dkl loss can't be computed using flow-based latent transforms.")
            elif self.latent_loss_type.lower() == 'mmd':
                pass  # Assigned by mother class
            elif self.latent_loss_type.lower() == 'logprob':
                self.latent_criterion = None  # unused - logprob directly implement in the latent loss method
            else:
                raise NotImplementedError("Unavailable latent loss '{}'".format(self.latent_loss_type))

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
        z_0_mu_logvar, z_0_sampled = self._encode_and_sample(x, sample_info)
        z_K_sampled, log_abs_det_jac = self.flow_transform(z_0_sampled)
        # TODO also get and return the style vector?
        w_style, x_out = self._decode_latent_vector(z_K_sampled)
        return z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out

    def latent_loss(self, z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac):
        if self.latent_loss_type.lower() == 'mmd':
            return super().latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)
        else:  # Flow-specific loss
            # log-probability of z_0 is evaluated knowing the gaussian distribution it was sampled from
            log_q_Z0_z0 = gaussian_log_probability(z_0_sampled, z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])
            # log-probability of z_K in the prior p_theta distribution
            # We model this prior as a zero-mean unit-variance multivariate gaussian
            log_p_theta_zK = standard_gaussian_log_probability(z_K_sampled)
            # Returned is the opposite of the ELBO terms
            if not self.normalize_losses:  # Default, which returns actual ELBO terms
                loss = -(log_p_theta_zK - log_q_Z0_z0 + log_abs_det_jac).mean()  # Mean over batch dimension
            else:  # Mean over batch dimension and latent vector dimension (D)
                loss = -(log_p_theta_zK - log_q_Z0_z0 + log_abs_det_jac).mean() / z_0_sampled.shape[1]
            return loss * self.latent_loss_compensation_factor

    def additional_latent_regularization_loss(self, z_0_mu_logvar):
        """ Returns an optional additional regularization loss, as configured using ctor args. """
        if self.latent_flow_input_reg_criterion is not None:
            return self.latent_flow_input_reg_criterion(z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :]) \
                   * self.latent_flow_input_regul_weight
        else:
            return torch.zeros((1, ), device=z_0_mu_logvar.device)
