"""
Base class to compute metrics about an abstract interpolation method
"""
import os.path
import pathlib
import pickle
import shutil
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, List, Any
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
import torch

from data.preset2d import Preset2d
from data.abstractbasedataset import PresetDataset
import evaluation.load
from evaluation.interpsequence import InterpSequence, LatentInterpSequence
from utils.timbretoolbox import InterpolationTimbreToolbox


class InterpBase(ABC):
    def __init__(self, num_steps=7, u_curve='linear', verbose=True):
        """
        Base attributes and methods of any interpolation engine.

        :param num_steps:
        :param u_curve: The type of curve used for the interpolation abscissa u.
        """
        self.num_steps = num_steps
        self.u_curve = u_curve
        self.verbose = verbose
        self.use_reduced_dataset = False  # faster debugging

    def get_u_interpolated(self):
        if self.u_curve == 'linear':
            return np.linspace(0.0, 1.0, self.num_steps, endpoint=True)
        elif self.u_curve == 'arcsin':
            return 0.5 + np.arcsin(np.linspace(-1.0, 1.0, self.num_steps, endpoint=True)) / np.pi
        elif self.u_curve == 'threshold':
            return (np.linspace(0.0, 1.0, self.num_steps) > 0.5).astype(float)
        else:
            raise NotImplementedError('Unimplemented curve {}'.format(self.u_curve))

    @property
    @abstractmethod
    def storage_path(self) -> pathlib.Path:
        pass

    def create_storage_directory(self):
        # First: create the dir to store data (erase any previously written eval files)
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)
        self.storage_path.mkdir(parents=True)
        if self.verbose:
            print("[{}] Results will be stored in '{}'".format(type(self).__name__, self.storage_path))

    @staticmethod
    def get_sequence_name(start_UID: int, end_UID: int, dataset: PresetDataset):
        start_name = dataset.get_name_from_preset_UID(start_UID)
        end_name = dataset.get_name_from_preset_UID(end_UID)
        return "[{}] '{}' --> [{}] '{}'".format(start_UID, start_name, end_UID, end_name)

    def try_process_dataset(self, force_re_eval=False):
        if force_re_eval:
            self.process_dataset()
        else:
            if os.path.exists(self.storage_path):
                print("[{}] Some results were already stored in '{}' - dataset won't be re-evaluated"
                      .format(type(self).__name__, self.storage_path))
            else:
                self.process_dataset()

    def process_dataset(self):
        self.render_audio()
        self.compute_and_save_interpolation_metrics()

    @abstractmethod
    def render_audio(self):
        pass

    def compute_and_save_interpolation_metrics(self):
        """
        Compute features for each individual audio file (which has already been rendered),
        then compute interpolation metrics for each sequence.
        """
        self.compute_store_timbre_toolbox_features()

        all_seqs_dfs = InterpolationTimbreToolbox.get_stored_postproc_sequences_descriptors(self.storage_path)
        features_stats = InterpolationTimbreToolbox.get_default_postproc_features_stats()
        interp_results = self._compute_interp_metrics(all_seqs_dfs, features_stats)
        with open(self.storage_path.joinpath('interp_results.pkl'), 'wb') as f:
            pickle.dump(interp_results, f)

    def compute_store_timbre_toolbox_features(self):
        _timbre_toolbox_path = '~/Documents/MATLAB/timbretoolbox'
        timbre_proc = InterpolationTimbreToolbox(
            _timbre_toolbox_path, self.storage_path, num_matlab_proc=12, remove_matlab_csv_after_usage=True)
        timbre_proc.run()
        timbre_proc.post_process_features(self.storage_path)

    @staticmethod
    def get_interp_results(storage_path: pathlib.Path):
        with open(storage_path.joinpath('interp_results.pkl'), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _compute_interp_metrics(all_seqs_dfs: List[pd.DataFrame], features_stats: Dict[str, Any]):
        """ Quantifies how smooth and linear the interpolation is, using previously computed audio features
        from timbretoolbox TODO maybe include some librosa features """
        # Compute interpolation performance for each interpolation metric, for each sequence
        interp_results = {'smoothness': list(), 'sum_squared_residuals': list()}
        for seq_df in all_seqs_dfs:
            seq_interp_results = InterpBase._compute_sequence_interp_metrics(seq_df, features_stats)
            for k in seq_interp_results:
                interp_results[k].append(seq_interp_results[k])
        # Then sum each interp metric up into a single df
        for k in interp_results:
            interp_results[k] = pd.DataFrame(interp_results[k])
        return interp_results

    @staticmethod
    def _compute_sequence_interp_metrics(seq: pd.DataFrame, features_stats: Dict[str, Any]):
        interp_metrics = {'smoothness': dict(), 'sum_squared_residuals': dict()}
        seq = seq.drop(columns="step_index")
        step_h = 1.0 / (len(seq) - 1.0)
        for col in seq.columns:
            feature_values = ((seq[col] - features_stats['mean'][col]) / features_stats['std'][col]).values
            # Smoothness: https://proceedings.neurips.cc/paper/2019/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf
            # Second-order central difference using a conv kernel, then compute the RMS of the smaller array
            smoothness = np.convolve(feature_values, [1.0, -2.0, 1.0], mode='valid') / (step_h ** 2)
            interp_metrics['smoothness'][col] = np.sqrt( (smoothness ** 2).mean() )
            # RSS
            target_linear_values = np.linspace(feature_values[0], feature_values[-1], num=feature_values.shape[0]).T
            interp_metrics['sum_squared_residuals'][col] = ((feature_values - target_linear_values) ** 2).sum()
        return interp_metrics


class NaivePresetInterpolation(InterpBase):
    def __init__(self, dataset, dataset_type, dataloader, base_storage_path: Union[str, pathlib.Path],
                 num_steps=7, u_curve='linear', verbose=True):
        super().__init__(num_steps, u_curve, verbose)
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.dataloader = dataloader
        self._base_storage_path = pathlib.Path(base_storage_path)

    @property
    def storage_path(self) -> pathlib.Path:
        return self._base_storage_path.joinpath('interp_{}'.format(self.dataset_type[0:5]))

    def render_audio(self):
        """ Generates interpolated sounds using the 'naïve' linear interpolation between VST preset parameters. """
        self.create_storage_directory()
        t_start = datetime.now()

        current_sequence_index = 0
        # Retrieve all latent vectors that will be used for interpolation
        # We assume that batch size is an even number...
        for batch_idx, minibatch in enumerate(self.dataloader):
            end_sequence_with_next_item = False  # If True, the next item is the 2nd (last) of an InterpSequence
            v_in, uid = minibatch[1], minibatch[2]
            for i in range(v_in.shape[0]):
                if self.verbose:
                    print("Processing item {}".format(current_sequence_index*2 + (i%2)))
                if not end_sequence_with_next_item:
                    end_sequence_with_next_item = True
                else:
                    # Compute and store interpolations one-by-one (gigabytes of audio data might not fit into RAM)
                    seq = InterpSequence(
                        self.storage_path, current_sequence_index, uid[i-1].item(), uid[i].item(),
                        self.get_sequence_name(uid[i-1].item(), uid[i].item(), self.dataset)
                    )
                    seq.u = self.get_u_interpolated()
                    # Convert learnable presets to VST presets  FIXME works for Dexed only
                    start_end_presets = list()
                    preset2d = Preset2d(self.dataset, learnable_tensor_preset=v_in[i-1])
                    start_end_presets.append(preset2d.to_raw())
                    preset2d = Preset2d(self.dataset, learnable_tensor_preset=v_in[i])
                    start_end_presets.append(preset2d.to_raw())
                    vst_interp_presets = self.get_interpolated_presets(seq.u, np.vstack(start_end_presets))
                    seq.audio, seq.spectrograms = self.generate_audio_and_spectrograms(vst_interp_presets)
                    seq.save()

                    current_sequence_index += 1
                    end_sequence_with_next_item = False
            if self.use_reduced_dataset:
                break
        if self.verbose:
            delta_t = (datetime.now() - t_start).total_seconds()
            print("[{}] Finished rendering audio for interpolations in {:.1f}min "
                  .format(type(self).__name__, delta_t / 60.0))

    def get_interpolated_presets(self, u: np.ndarray, start_end_presets: np.ndarray):
        interp_f = scipy.interpolate.interp1d(
            [0.0, 1.0], start_end_presets, kind='linear', axis=0,
            bounds_error=True)  # extrapolation disabled, no fill_value
        return interp_f(u)

    def generate_audio_and_spectrograms(self, vst_presets: np.ndarray):
        midi_pitch, midi_vel = self.dataset.default_midi_note
        audio_renders = [self.dataset._render_audio(vst_presets[i, :], midi_pitch, midi_vel)
                         for i in range(vst_presets.shape[0])]
        spectrograms = [self.dataset.compute_spectrogram(a[0]) for a in audio_renders]
        return audio_renders, spectrograms


class ModelBasedInterpolation(InterpBase):
    def __init__(
            self, model_loader: Optional[evaluation.load.ModelLoader] = None, device='cpu', num_steps=7,
            u_curve='linear', latent_interp_kind='linear', verbose=True,
            storage_path: Optional[pathlib.Path] = None
    ):
        """
        A class for performing interpolations using a neural network model whose inputs are latent vectors.

        :param model_loader: If given, most of other arguments (related to the model and corresponding
         dataset) will be ignored.
        """
        super().__init__(num_steps=num_steps, verbose=verbose, u_curve=u_curve)
        self.latent_interp_kind = latent_interp_kind
        self._storage_path = storage_path
        if model_loader is not None:
            self._model_loader = model_loader
            self.device = model_loader.device
            self.dataset = model_loader.dataset
            self.dataset_type = model_loader.dataset_type
            self.dataloader, self.dataloader_num_items = model_loader.dataloader, model_loader.dataloader_num_items
            self.ae_model = model_loader.ae_model
        else:
            self.device = device
            self.dataset, self.dataset_type, self.dataloader, self.dataloader_num_items = None, None, None, None

    @property
    def storage_path(self) -> pathlib.Path:
        return self._storage_path

    def render_audio(self):
        """ Performs an interpolation over the whole given dataset (usually validation or test), using pairs
        of items from the dataloader. Dataloader should be deterministic. Total number of interpolations computed:
        len(dataloder) // 2. """
        self.create_storage_directory()
        t_start = datetime.now()

        # to store all latent-specific stats (each sequence is written to SSD before computing the next one)
        # encoded latent vectors (usually different from endpoints, if the corresponding preset is not 100% accurate)
        z_ae = list()
        # Interpolation endpoints
        z_endpoints = list()

        current_sequence_index = 0
        # Retrieve all latent vectors that will be used for interpolation
        # We assume that batch size is an even number...
        for batch_idx, minibatch in enumerate(self.dataloader):
            end_sequence_with_next_item = False  # If True, the next item is the 2nd (last) of an InterpSequence
            x_in, v_in, uid, notes, label = [m for m in minibatch]
            N = x_in.shape[0]
            for i in range(N):
                if not end_sequence_with_next_item:
                    if self.verbose:
                        print("Item {} (mini-batch {}/{})".format(i + batch_idx * N, batch_idx+1, len(self.dataloader)))
                    end_sequence_with_next_item = True
                else:
                    # It's easier to compute interpolations one-by-one (all data might not fit into RAM)
                    seq = LatentInterpSequence(
                        self.storage_path, current_sequence_index, uid[i-1].item(), uid[i].item(),
                        self.get_sequence_name(uid[i-1].item(), uid[i].item(), self.dataset)
                    )
                    z_start, z_start_first_guess, acc, L1_err = self.compute_latent_vector(
                        x_in[i-1:i], v_in[i-1:i], uid[i-1:i], notes[i-1:i])
                    if acc < 100.0 or L1_err > 0.0:
                        warnings.warn("UID {}: acc={:.2f}, l1err={:.2f}".format(uid[i-1].item(), acc, L1_err))
                        # raise AssertionError(acc, L1_err)
                    z_end, z_end_first_guess, acc, L1_err = self.compute_latent_vector(
                        x_in[i:i+1], v_in[i:i+1], uid[i:i+1], notes[i:i+1])
                    if acc < 100.0 or L1_err > 0.0:
                        warnings.warn("UID {}: acc={:.2f}, l1err={:.2f}".format(uid[i].item(), acc, L1_err))
                        # raise AssertionError(acc, L1_err)
                    z_ae.append(z_start_first_guess), z_ae.append(z_end_first_guess)
                    z_endpoints.append(z_start), z_endpoints.append(z_end)

                    seq.u, seq.z = self.interpolate_latent(z_start, z_end)
                    seq.audio, seq.spectrograms = self.generate_audio_and_spectrograms(seq.z)
                    seq.save()

                    current_sequence_index += 1
                    end_sequence_with_next_item = False
            if self.use_reduced_dataset:
                break

        # TODO also store "interp hparams" which were used to compute interpolation (e.g. u_curve, inverse log prob, ...)

        all_z_ae = torch.vstack(z_ae).detach().clone().cpu().numpy()
        all_z_endpoints = torch.vstack(z_endpoints).detach().clone().cpu().numpy()
        with open(self.storage_path.joinpath('all_z_ae.np.pkl'), 'wb') as f:
            pickle.dump(all_z_ae, f)
        with open(self.storage_path.joinpath('all_z_endpoints.np.pkl'), 'wb') as f:
            pickle.dump(all_z_endpoints, f)
        if self.verbose:
            delta_t = (datetime.now() - t_start).total_seconds()
            print("[{}] Finished rendering audio for {} interpolations in {:.1f}min total ({:.1f}s / interpolation)"
                  .format(type(self).__name__, all_z_endpoints.shape[0] // 2, delta_t / 60.0,
                          delta_t / (all_z_endpoints.shape[0] // 2)))

    @abstractmethod
    def compute_latent_vector(self, x_in, v_in, uid, notes) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """ Computes the most appropriate latent vector (child class implements this method)

        :returns: z_estimated, z_first_guess, preset_accuracy, preset_L1_error
        """
        pass

    def interpolate_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor]:
        """ Returns a N x D tensor of interpolated latent vectors, where N is the number of interpolation steps (here:
        considered as a batch size) and D is the latent dimension. Each latent coordinate is interpolated independently.

        Non-differentiable: based on scipy.interpolate.interp1d.

        :param z_start: 1 x D tensor
        :param z_end: 1 x D tensor
        :returns: u, interpolated_z
        """
        # TODO allow EXTRAPOLATION
        z_cat = torch.cat([z_start, z_end], dim=0)
        # TODO SPHERICAL interpolation available?
        interp_f = scipy.interpolate.interp1d(
            [0.0, 1.0], z_cat.clone().detach().cpu().numpy(), kind=self.latent_interp_kind, axis=0,
            bounds_error=True)  # extrapolation disabled, no fill_value
        u_interpolated = self.get_u_interpolated()
        z_interpolated = interp_f(u_interpolated)
        return u_interpolated, torch.tensor(z_interpolated, device=self.device, dtype=torch.float32)

    @abstractmethod
    def generate_audio_and_spectrograms(self, z: torch.Tensor):
        """ Returns a list of audio waveforms and/or a list of spectrogram corresponding to latent vectors z (given
            as a 2D mini-batch of vectors). """
        pass



if __name__ == "__main__":
    _storage_path = '/media/gwendal/Data/Interpolations/LinearNaive/interp_validation'
    #_storage_path = '/media/gwendal/Data/Interpolations/ThresholdNaive/interp_validation'cd ../
    _results1 = InterpBase.get_interp_results(pathlib.Path(_storage_path))

    _storage_path = '/media/gwendal/Data/Logs/preset-vae/presetAE/combined_vae_beta1.60e-04_presetfactor0.20/interp_validation'
    _results2 = InterpBase.get_interp_results(pathlib.Path(_storage_path))

    _all_seqs_dfs = InterpolationTimbreToolbox.get_stored_postproc_sequences_descriptors(_storage_path)
    _features_stats =  InterpolationTimbreToolbox.get_default_postproc_features_stats()
    _results_dfs = InterpBase._compute_interp_metrics(_all_seqs_dfs, _features_stats)


