"""
Classes to generate and evaluate interpolations between samples.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import scipy.interpolate
import torch

import data.abstractbasedataset
import data.build
import logs.logger
import model.build
import utils.config

import evaluation.load
import evaluation.interpbase



class LatentInterpolation(evaluation.interpbase.ModelBasedInterpolation):
    def __init__(self, model_loader: Optional[evaluation.load.ModelLoader] = None,
                 num_steps=7, device='cpu',
                 latent_interp='linear',
                 generator=None, init_generator=True):
        """
        Generic class for interpolating latent vectors (between two reference latent vectors), and for generating
        samples (spectrograms and/or audio) from those interpolated latent vectors.
        The generator

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

    def compute_latent_vector(self, x, sample_info, v_target) -> Tuple[torch.Tensor, float, float]:
        raise NotImplementedError()

    def interpolate_spectrograms_from_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """ Returns u, z, and a N x 1 x H x W tensor of interpolated spectrograms, using the latent representations
            of those spectrograms as interpolation inputs. """
        # FIXME return lists only, and handle both audio and spectrogram outputs
        u, z = self.interpolate_latent(z_start, z_end)
        return u, z, self.gen.generate_from_latent_vector(z)


# TODO don't inherit from LatentInterpolation
class SynthPresetLatentInterpolation(LatentInterpolation):
    def __init__(self, model_loader: evaluation.load.ModelLoader, num_steps=7,
                 latent_interp='linear'):
        super().__init__(model_loader=model_loader, num_steps=num_steps,
                         init_generator=False, latent_interp=latent_interp)
        if not isinstance(self.dataset, data.abstractbasedataset.PresetDataset):
            raise NotImplementedError("This evaluation class is available for a PresetDataset only (current "
                                      "self.dataset type: {})".format(type(self.dataset)))
        if self.extended_ae_model is None or self.reg_model is None:
            raise AssertionError("A full model (including a regression network) must be loaded.")

        # TODO finish all checks

    def compute_latent_vector(self, x, sample_info, v_target) -> Tuple[torch.Tensor, float, float]:
        return self.find_preset_inverse(x, sample_info, v_target)

    def find_preset_inverse(self, x_in, sample_info, u_target):
        self.ae_model.eval()
        with torch.no_grad():
            ae_outputs = self.ae_model(x_in, sample_info)
            z_first_guess = ae_outputs[2]
        self.reg_model.eval()  # Dropout must be de-activated
        # TODO proper ctor arg to use flow inverse, or not
        return self.reg_model.find_preset_inverse(u_target, z_first_guess)  # Flow inverse
        # return self.reg_model.find_preset_inverse_SGD(u_target, z_first_guess)  # SGD

    @property
    def gen(self):
        # TODO apply reg model and generate sound+spectrogram
        return None


if __name__ == "__main__":
    _device = 'cpu'
    _model_loader = evaluation.load.ModelLoader("saved/FlowReg_dimz5020/dequantized_out_L2_loss__permsFalse",
                                                _device, 'validation')

    preset_interpolator = SynthPresetLatentInterpolation(_model_loader)
    preset_interpolator.process_dataset()

