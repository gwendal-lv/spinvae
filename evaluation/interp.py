"""
Classes to generate and evaluate interpolations between samples.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.interpolate
import torch

import data.abstractbasedataset
import data.build
import evaluation.load
import logs.logger
import model.build
import utils.config


class LatentInterpolationEval:
    def __init__(self, generator, num_steps=7, device='cpu', kind='linear'):
        """

        :param generator: Any model that provides a generate_from_latent_vector(z) method (where z is a batch of
            vectors), a .dim_z attribute. E.g. an ae_model.
        :param kind: See https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.interp1d.html
        """
        self.gen = generator
        self.num_steps = num_steps
        self.device = device
        self.kind = kind

    @property
    def dim_z(self):
        return self.gen.dim_z

    def interpolate_spectrograms_from_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """ Returns u, z, and a N x 1 x H x W tensor of interpolated spectrograms, using the latent representations
            of those spectrograms as interpolation inputs. """
        u, z = self.interpolate_latent(z_start, z_end)
        return u, z, self.gen.generate_from_latent_vector(z)

    def interpolate_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor]:
        """ Returns a N x D tensor of interpolated latent vectors, where N is the number of interpolation steps (here:
        considered as a batch size) and D is the latent dimension. Each latent coordinate is interpolated independently.

        Non-differentiable: based on scipy.interpolate.interp1d.

        :param z_start: 1 x D tensor
        :param z_end: 1 x D tensor
        :returns: u, interpolated_z
        """
        # TODO try RBF interpolation instead of interp 1d ?
        z_cat = torch.cat([z_start, z_end], dim=0)
        interp_f = scipy.interpolate.interp1d([0.0, 1.0], z_cat.clone().detach().cpu().numpy(), kind=self.kind, axis=0,
                                              bounds_error=True)  # extrapolation disabled, no fill_value
        u_interpolated = np.linspace(0.0, 1.0, self.num_steps, endpoint=True)
        return u_interpolated, torch.tensor(interp_f(u_interpolated), device=self.device, dtype=torch.float32)

    # TODO generate and save independent figure (easier to show in HTML tables // github pages)

    # TODO compute 'interpolation smoothness' coefficients


class PresetInterpolatorEval:
    def __init__(self, model_loader: evaluation.load.ModelLoader):
        self.dataset = model_loader.dataset
        if not isinstance(self.dataset, data.abstractbasedataset.PresetDataset):
            raise NotImplementedError("This evaluation class is available for a PresetDataset only.")
        self.dataloader, dataloaders_nb_items \
            = data.build.get_split_dataloaders(model_loader.train_config, self.dataset, num_workers=0)
        self.extended_ae_model, self.ae_model, self.reg_model \
            = model_loader.extended_ae_model, model_loader.ae_model, model_loader.reg_model
        if self.extended_ae_model is None or self.reg_model is None:
            raise AssertionError("A full model (including a regression network) must be loaded.")

    def find_preset_inverse(self, x_in, sample_info, u_target):
        self.ae_model.eval()
        ae_outputs = self.ae_model(x_in, sample_info)
        z_first_guess = ae_outputs[2]
        self.reg_model.eval()  # Dropout must be de-activated
        # TODO search element-by-element
        z_estimated, acc, num_loss = self.reg_model.find_preset_inverse(u_target, z_first_guess)
        z_diff = z_first_guess - z_estimated
        print(z_first_guess - z_estimated, acc, num_loss)
        print("max z diff = {}".format(torch.max(torch.abs(z_diff))))


if __name__ == "__main__":
    _device = 'cpu'
    _model_loader = evaluation.load.ModelLoader("saved/ControlsRegr_allascat/htanhTrue_softmT°0.2_permTrue", _device)

    preset_interpolator = PresetInterpolatorEval(_model_loader)
    dataloader_iter = iter(preset_interpolator.dataloader['test'])

    items = next(dataloader_iter)
    r = range(0, 1)
    _x_in = items[0][r, :, :, :]
    _u_target = items[1][r, :]
    _sample_info = items[2][r, :]
    preset_interpolator.find_preset_inverse(_x_in, _sample_info, _u_target)
    print('OK')
