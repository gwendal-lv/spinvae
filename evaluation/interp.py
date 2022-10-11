"""
Classes to generate and evaluate interpolations between samples.
"""
import pathlib
import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import scipy.interpolate
import torch

import data.abstractbasedataset
import data.build
import utils.probability
from data.preset2d import Preset2dHelper, Preset2d

import evaluation.load
import evaluation.interpbase



class LatentInterpolation(evaluation.interpbase.ModelBasedInterpolation):
    def __init__(self, model_loader: Optional[evaluation.load.ModelLoader] = None,
                 num_steps=7, device='cpu',
                 latent_interp='linear',
                 generator=None, init_generator=True):
        """
        Generic class for interpolating latent vectors (between two reference latent vectors), and for generating
        samples (spectrograms and/or audio) from those interpolated latent vectors using a generator model.

        :param generator: Any model that provides a generate_from_latent_vector(z) method (where z is a batch of
              vectors), a .dim_z attribute (e.g. a decoder model). Using this argument will prevent this class from
              using a given model_loader.
              TODO handle audio generators
        :param model_loader: ModelLoader instance to reload a trained AE model. Default behavior: ae_model.decoder
            will be used as self.gen generative model.
        :param init_generator: If True, the generator will be automatically set to ae_model.decoder from the
            loaded model. Otherwise, the generator should be set by a child class that inherited this class.
        :param latent_interp: See https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.interp1d.html
        """
        # different init if generator is None or not
        if generator is not None:
            super().__init__(model_loader=None, device=device, num_steps=num_steps, latent_interp_kind=latent_interp)
            self._gen = generator
            if not init_generator:
                raise ValueError("init_generator is False but a generator ctor arg was given.")
        else:
            super().__init__(model_loader=model_loader, device=device, num_steps=num_steps,
                             latent_interp_kind=latent_interp)
            if init_generator:
                if self.ae_model is None:
                    raise AssertionError("self.ae_model was not set - it cannot be used as a generator for interp.")
                self._gen = self.ae_model
            else:
                self._gen = None

    @property
    def gen(self):
        return self._gen

    @property
    def dim_z(self):
        return self._gen.dim_z

    # FIXME prototype not compatible anymore with the new datasets (midi notes given as a separate tensor)
    def compute_latent_vector(self, x, sample_info, v_target) -> Tuple[torch.Tensor, float, float]:
        raise NotImplementedError()

    def interpolate_spectrograms_from_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """ Returns u, z, and a N x 1 x H x W tensor of interpolated spectrograms, using the latent representations
            of those spectrograms as interpolation inputs. """
        # FIXME return lists only, and handle both audio and spectrogram outputs
        u, z = self.interpolate_latent(z_start, z_end)
        return u, z, self.gen.generate_from_latent_vector(z)

    def generate_audio_and_spectrograms(self, z: torch.Tensor):
        raise NotImplementedError()


class SynthPresetLatentInterpolation(evaluation.interpbase.ModelBasedInterpolation):
    def __init__(self, model_loader: evaluation.load.ModelLoader, num_steps=7,
                 u_curve='linear', latent_interp='linear', refine_level=0,
                 storage_path: Optional[pathlib.Path] = None, reference_storage_path: Optional[pathlib.Path] = None):
        """

        :param refine_level: The amount of refinement applied to the original inferred latent codes, in order
            to try to improve the presets' reconstruction accuracy.
        """
        super().__init__(
            model_loader=model_loader, num_steps=num_steps, u_curve=u_curve, latent_interp_kind=latent_interp,
            storage_path=storage_path, reference_storage_path=reference_storage_path
        )
        self.refine_level = refine_level
        if not isinstance(self.dataset, data.abstractbasedataset.PresetDataset):
            raise NotImplementedError("This evaluation class is available for a PresetDataset only (current "
                                      "self.dataset type: {})".format(type(self.dataset)))

        # TODO finish all checks

    def _init_nn_model(self):
        pass

    def compute_latent_vector(self, x_in, v_in, uid, notes):
        self.ae_model.eval()
        with torch.no_grad():
            ae_out = self.ae_model(x_in, v_in, uid, notes)
            ae_out = self.ae_model.parse_outputs(ae_out)
            # Presets need to be flattened for u, z interpolation, then un-flattened during audio interp/generation
            z_first_guess_flat = self.ae_model.flatten_latent_values(ae_out.z_sampled)
            # We'll keep those as a reference
            z_mu_first_guess_flat = self.ae_model.flatten_latent_values(ae_out.z_mu)
            z_var_first_guess_flat = self.ae_model.flatten_latent_values(ae_out.z_var)
            z_logvar_first_guess_flat = torch.log(z_var_first_guess_flat)

        num_l1_error, acc = ae_out.u_l1_error.item(), ae_out.u_accuracy.item()
        num_l1_error_first_guess, acc_first_guess = num_l1_error, acc
        z_estimated = z_first_guess_flat  # Default behavior, this tensor might be modified during optimization
        # TODO as args: threshold acc/L1
        acc_th, num_l1_th = 0.99, 0.01

        if self.refine_level > 0:
            warnings.warn("'z refinement' is experimental")
            if self.refine_level == 1:
                lr, n_steps, z_NLL_factor = 0.1, 50, 0.1
            elif self.refine_level == 2:
                lr, n_steps, z_NLL_factor = 1.0, 50, 0.01
            else:
                raise ValueError(self.refine_level)  # Should be 1 or 2 (if not 0)

            if num_l1_error > num_l1_th or acc < acc_th:  # we optimize only if necessary
                #  we'll optimize a small z delta to find a better z (that reconstructs the preset better)
                #    A Tensor will work, torch.autograd.Variable is deprecated
                z_delta = torch.zeros_like(z_first_guess_flat, device=z_first_guess_flat.device, requires_grad=True)
                optimizer = torch.optim.SGD([z_delta], lr=lr, momentum=0.5)
                for step in range(n_steps):  # TODO max n steps as arg
                    optimizer.zero_grad()
                    self.ae_model.zero_grad()  # Needs to be done manually - not in the optimizer
                    z_flat = z_first_guess_flat + z_delta
                    z_multi_level = self.ae_model.unflatten_latent_values(z_flat)
                    dec_out = self.ae_model.decoder.preset_decoder(z_multi_level, u_target=v_in)
                    u_out, u_numerical_nll, u_categorical_nll, num_l1_error, acc = dec_out

                    # Stop optimization as soon as possible
                    # TODO raise warning if results worsen...
                    if num_l1_error.item() < num_l1_th and acc.item() > acc_th:
                        break

                    # TODO compute log-prob under the VAE-encoded gaussian distrib q(z | x, v)
                    # Ensures that latent codes remains highly log-probable under the posterior distribution
                    z_NLL_initial_distribution = - utils.probability.gaussian_log_probability(
                        z_flat, z_mu_first_guess_flat, z_logvar_first_guess_flat, add_log_2pi_term=True)
                    z_NLL_initial_distribution = z_NLL_initial_distribution / z_flat.shape[1]

                    # TODO maybe decrease the loss for perfectly-reconstructed params?
                    # Note: 1.0 z_NLL factor leads to deteriorated results (too strong gradients?)
                    loss = u_numerical_nll + u_categorical_nll + z_NLL_factor * z_NLL_initial_distribution
                    loss.backward()
                    optimizer.step()
                acc, num_l1_error = acc.item(), num_l1_error.item()
                z_delta.requires_grad = False
                z_estimated = z_first_guess_flat + z_delta
                if self.verbose:  # TODO improve prints
                    print("opt ended at step {} - acc = {:.1f}%    num_l1_err = {:.3f}    avg |delta z| = {}    max |delta z| = {}"
                          .format(step, acc * 100.0, num_l1_error, z_delta.abs().mean().item(), z_delta.abs().max().item()))
                    print("Improvement:    acc {}    num_l1_err {}"
                          .format((acc - acc_first_guess) * 100.0, num_l1_error - num_l1_error_first_guess))
            else:
                if self.verbose:
                    print("NO optimization necessary")

        return z_estimated, z_first_guess_flat, acc, num_l1_error, acc_first_guess, num_l1_error_first_guess


    @property
    def gen(self):
        raise NotImplementedError()

    def generate_audio_and_spectrograms(self, z: torch.Tensor):
        z_multi_level = self.ae_model.unflatten_latent_values(z)
        # Apply model to each step retrieve corresponding presets
        decoder_out = self.ae_model.decoder.preset_decoder(z_multi_level, u_target=None)
        v_out = decoder_out[0]
        # Convert learnable presets to VST presets - non-batched code (processes individual presets)
        vst_presets = list()
        for i in range(v_out.shape[0]):
            preset2d = Preset2d(self.dataset, learnable_tensor_preset=v_out[i])
            vst_presets.append(preset2d.to_raw())
        midi_pitch, midi_vel = self.dataset.default_midi_note
        audio_renders = [self.dataset._render_audio(raw_preset, midi_pitch, midi_vel) for raw_preset in vst_presets]
        spectrograms = [self.dataset.compute_spectrogram(audio_wav[0]) for audio_wav in audio_renders]
        return audio_renders, spectrograms

