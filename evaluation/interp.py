"""
Classes to generate and evaluate interpolations between samples.
"""
import pathlib
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import scipy.interpolate
import torch

import data.abstractbasedataset
import data.build
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
                 u_curve='linear', latent_interp='linear',
                 storage_path: Optional[pathlib.Path] = None, reference_storage_path: Optional[pathlib.Path] = None):
        super().__init__(
            model_loader=model_loader, num_steps=num_steps, u_curve=u_curve, latent_interp_kind=latent_interp,
            storage_path=storage_path, reference_storage_path=reference_storage_path
        )
        if not isinstance(self.dataset, data.abstractbasedataset.PresetDataset):
            raise NotImplementedError("This evaluation class is available for a PresetDataset only (current "
                                      "self.dataset type: {})".format(type(self.dataset)))

        # TODO finish all checks

    def compute_latent_vector(self, x_in, v_in, uid, notes):
        self.ae_model.eval()
        with torch.no_grad():
            ae_out = self.ae_model(x_in, v_in, uid, notes)
            ae_out = self.ae_model.parse_outputs(ae_out)
        # Presets need to be flattened for u, z interpolation, then un-flattened during audio interpolation/generation
        z_first_guess = self.ae_model.flatten_latent_values(ae_out.z_sampled)
        # return self.find_preset_inverse(x_in, v_in, uid, notes) FIXME re-implement
        z_estimated = z_first_guess  # FIXME
        return z_estimated, z_first_guess, ae_out.u_accuracy.item(), ae_out.u_l1_error.item()

    def find_preset_inverse(self, x_in, v_in, uid, notes):
        """ TODO doc """
        assert x_in.shape[0] == v_in.shape[0] == uid.shape[0] == notes.shape[0] == 1
        self.ae_model.eval()
        with torch.no_grad():
            ae_out = self.ae_model(x_in, v_in, uid, notes)
            ae_out = self.ae_model.parse_outputs(ae_out)
        # TODO proper ctor arg to use flow inverse, or not
        # return self.reg_model.find_preset_inverse(u_target, z_first_guess)  # Flow inverse
        raise NotImplementedError("")  # TODO find exact preset inverse
        return None  # self.reg_model.find_preset_inverse_SGD(u_target, z_0_sampled, z_0_mu_logvar)

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


if __name__ == "__main__":
    _device = 'cpu'
    _root_path = Path(__file__).resolve().parent.parent
    _model_path = _root_path.joinpath('../Data_SSD/Logs/preset-vae/presetAE/combined_vae_beta1.60e-04_presetfactor0.50')
    _model_loader = evaluation.load.ModelLoader(_model_path, _device, 'validation')

    _num_steps = 9

    if True:  # TODO Gen naive interpolations ? (LINEAR)
        naive_preset_interpolator = evaluation.interpbase.NaivePresetInterpolation(
            _model_loader.dataset, _model_loader.dataset_type, _model_loader.dataloader,
            _root_path.parent.joinpath('Data_SSD/Interpolations/LinearNaive'), num_steps=_num_steps)
        naive_preset_interpolator.process_dataset()

    # TODO additional path suffix for different interp hparams
    if True:
        preset_interpolator = SynthPresetLatentInterpolation(
            _model_loader, num_steps=_num_steps, u_curve='linear', latent_interp='linear',
        )
        preset_interpolator.process_dataset()

