"""
Base class to compute metrics about an abstract interpolation method
"""
import os.path
import pathlib
import pickle
import shutil
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
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
        self.all_librosa_interp_metrics = dict()

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
        os.mkdir(self.storage_path)
        if self.verbose:
            print("[{}] Results will be stored in '{}'".format(type(self).__name__, self.storage_path))

    @staticmethod
    def get_sequence_name(start_UID: int, end_UID: int, dataset: PresetDataset):
        start_name = dataset.get_name_from_preset_UID(start_UID)
        end_name = dataset.get_name_from_preset_UID(end_UID)
        return "[{}] '{}' --> [{}] '{}'".format(start_UID, start_name, end_UID, end_name)

    def librosa_interp_metrics_init(self):
        self.all_librosa_interp_metrics = None

    def append_librosa_interp_metric(self, m: InterpSequence):
        """ Internally stores all available librosa metrics (might be None) from the given sequence. """
        if m.librosa_interpolation_metrics is not None:
            if self.all_librosa_interp_metrics is None:  # Create dict and init lists of arrays
                self.all_librosa_interp_metrics = {k: list() for k in m.librosa_interpolation_metrics}
            for k, v in m.librosa_interpolation_metrics.items():
                self.all_librosa_interp_metrics[k].append(v)

    def store_librosa_interp_metrics(self):
        if self.all_librosa_interp_metrics is not None:
            stacked_metrics = {k: np.stack(l, axis=0) for k, l in self.all_librosa_interp_metrics.items()}
            with open(self.storage_path.joinpath('all_librosa_interp_metrics.pkl'), 'wb') as f:
                pickle.dump(stacked_metrics, f)
        else:
            warnings.warn("Cannot store librosa metrics (which are None).")

    # TODO compute TT interp metrics


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
        return self._base_storage_path.joinpath('interp_{}'.format(self.dataset_type))

    def process_dataset(self):
        """ Generates interpolated sounds using the 'naïve' linear interpolation between VST preset parameters. """
        self.create_storage_directory()
        t_start = datetime.now()

        self.librosa_interp_metrics_init()

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
                    seq = InterpSequence(self.storage_path, current_sequence_index)
                    seq.UID_start, seq.UID_end = uid[i-1].item(), uid[i].item()
                    seq.name = self.get_sequence_name(seq.UID_start, seq.UID_end, self.dataset)
                    seq.u = self.get_u_interpolated()
                    # Convert learnable presets to VST presets  FIXME works for Dexed only
                    start_end_presets = list()
                    preset2d = Preset2d(self.dataset, learnable_tensor_preset=v_in[i-1])
                    start_end_presets.append(preset2d.to_raw())
                    preset2d = Preset2d(self.dataset, learnable_tensor_preset=v_in[i])
                    start_end_presets.append(preset2d.to_raw())
                    vst_interp_presets = self.get_interpolated_presets(seq.u, np.vstack(start_end_presets))
                    seq.audio, seq.spectrograms = self.generate_audio_and_spectrograms(vst_interp_presets)
                    seq.process_and_save()
                    self.append_librosa_interp_metric(seq)

                    current_sequence_index += 1
                    end_sequence_with_next_item = False
            if batch_idx >= 1:  # FIXME, TEMP
                break
        self.store_librosa_interp_metrics()
        if self.verbose:
            delta_t = (datetime.now() - t_start).total_seconds()
            print("[{}] Finished processing interpolations in {:.1f}min ".format(type(self).__name__, delta_t / 60.0))

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
    def __init__(self, model_loader: Optional[evaluation.load.ModelLoader] = None,
                 device='cpu', num_steps=7,
                 u_curve='linear', latent_interp_kind='linear', verbose=True):
        """
        A class for performing interpolations using a neural network model whose inputs are latent vectors.

        :param model_loader: If given, most of other arguments (related to the model and corresponding
         dataset) will be ignored.
        """
        super().__init__(num_steps=num_steps, verbose=verbose, u_curve=u_curve)
        self.latent_interp_kind = latent_interp_kind
        if model_loader is not None:
            self._model_loader = model_loader
            self.device = model_loader.device
            self.dataset = model_loader.dataset
            self.dataset_type = model_loader.dataset_type
            self.dataloader, self.dataloader_num_items = model_loader.dataloader, model_loader.dataloader_num_items
            self._storage_path = model_loader.path_to_model_dir
            self.ae_model = model_loader.ae_model
        else:
            self.device = device
            self.dataset, self.dataset_type, self.dataloader, self.dataloader_num_items = None, None, None, None
            self._storage_path = None

    @property
    def storage_path(self) -> pathlib.Path:
        if self._storage_path is not None:
            return self._storage_path.joinpath('interp_{}'.format(self.dataset_type))
        else:
            raise AssertionError("self._storage_path was not set (this instance was not built using a ModelLoader)")

    def process_dataset(self):
        """ Performs an interpolation over the whole given dataset (usually validation or test), using pairs
        of items from the dataloader. Dataloader should be deterministic. Total number of interpolations computed:
        len(dataloder) // 2. """
        self.create_storage_directory()
        t_start = datetime.now()

        self.librosa_interp_metrics_init()
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
                        print("Item {} (mini-batch {}/{})"
                              .format(i + batch_idx * N, batch_idx+1, len(self.dataloader)))
                    end_sequence_with_next_item = True
                else:
                    # It's easier to compute interpolations one-by-one (all data might not fit into RAM)
                    seq = LatentInterpSequence(self.storage_path, current_sequence_index)
                    seq.UID_start, seq.UID_end = uid[i-1].item(), uid[i].item()
                    seq.name = self.get_sequence_name(seq.UID_start, seq.UID_end, self.dataset)
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
                    seq.process_and_save()
                    self.append_librosa_interp_metric(seq)

                    current_sequence_index += 1
                    end_sequence_with_next_item = False

            # FIXME TEMP
            if batch_idx >= 1:
                break

        self.store_librosa_interp_metrics()

        # TODO also store "interp hparams" which were used to compute interpolation (e.g. u_curve, inverse log prob, ...)

        all_z_ae = torch.vstack(z_ae).detach().clone().cpu().numpy()
        all_z_endpoints = torch.vstack(z_endpoints).detach().clone().cpu().numpy()
        with open(self.storage_path.joinpath('all_z_ae.np.pkl'), 'wb') as f:
            pickle.dump(all_z_ae, f)
        with open(self.storage_path.joinpath('all_z_endpoints.np.pkl'), 'wb') as f:
            pickle.dump(all_z_endpoints, f)
        if self.verbose:
            delta_t = (datetime.now() - t_start).total_seconds()
            print("[{}] Finished processing {} interpolations in {:.1f}min total ({:.1f}s / interpolation)"
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


def _compute_metric_statistics(flat_metric: np.ndarray, metric_name: str):
    """
    :returns: FIXME DOC sum_squared_normalized_residuals, linregress_R2, pearson_r_squared
    """
    # TODO acceleration RMS (smoothness factor)
    # First, we perform linear regression on the output features
    linregressions = [scipy.stats.linregress(np.linspace(0.0, 1.0, num=y.shape[0]), y) for y in flat_metric]
    linregr_R2 = np.asarray([l.rvalue**2 for l in linregressions])
    linregr_pvalues = np.asarray([l.pvalue for l in linregressions])
    # target features (ideal linear increase/decrease)
    target_metric = np.linspace(flat_metric[:, 0], flat_metric[:, -1], num=flat_metric.shape[1]).T
    target_amplitudes = flat_metric.max(axis=1) - flat_metric.min(axis=1)
    mean_target_amplitude = target_amplitudes.mean()
    target_max_abs_values = np.abs(flat_metric).max(axis=1)
    target_relative_variation = target_amplitudes / target_max_abs_values
    # Sum of squared (normalized) residuals
    residuals = (flat_metric - target_metric) / mean_target_amplitude
    # r2 pearson correlation - might trigger warnings if an array is (too) constant. Dismiss p-values
    pearson_r = np.asarray(
        [scipy.stats.pearsonr(target_metric[i, :], flat_metric[i, :])[0] for i in range(target_metric.shape[0])])
    pearson_r = pearson_r[~np.isnan(pearson_r)]
    # max of abs derivative (compensated for step size, normalized vs. target amplitude)
    diffs = np.diff(flat_metric, axis=1) * flat_metric.shape[1] / mean_target_amplitude
    max_abs_diffs = np.abs(diffs.max(axis=1))  # Biggest delta for all steps, average for all sequences
    return {'metric': metric_name,
            'sum_squared_residuals': (residuals ** 2).mean(),
            'linregression_R2': linregr_R2.mean(),
            'target_pearson_r2': (pearson_r**2).mean(),  # Identical to R2 when there is no nan r value
            'max_abs_delta': max_abs_diffs.mean()
            }


def process_features(storage_path: pathlib.Path):
    """ Reads the 'all_interp_metrics.pkl' file (pre-computed audio features at each interp step) and computes:
        - constrained affine regression values (start and end values are start/end feature values). They can't be used
          to compute a R2 coeff of determination, because some values -> -infty (when target is close to constant)
        - R2 coefficients for each regression curve fitted to interpolated features (not the GT ideal features)
    TODO where to store results?? Dataframe?
    """
    with open(storage_path.joinpath('all_interp_metrics.pkl'), 'rb') as f:
        all_interp_metrics = pickle.load(f)
    # TODO pre-preprocessing of '_full' metrics: compute their avg and std as new keys
    # Compute various stats  for all metrics - store all in a dataframe
    results_list = list()
    for k, metric in all_interp_metrics.items():
        # sklearn r2 score (coefficient of determination) expects inputs shape: (n_samples, n_outputs)
        #    We can't use it with constrained regressions (from start to end point) because R2 values
        #    become arbitrarily negative for flat target data (zero-slope)
        # So: we'll use the usual rvalue from linear regression, where R2 (determination coeff) = rvalue^2 >= 0.0
        #    (pvalue > 0.05 indicates that we can't reject H0: "the slope is zero")
        # TODO also compute sum of errors compared to the "ideal start->end affine regression"
        if len(metric.shape) == 2:  # scalar metric (real value for each step of each interpolation sequence)
            results_list.append(_compute_metric_statistics(metric, k))
        elif len(metric.shape) == 3:  # vector metric (for each step of each sequence) e.g. RMS values for time frames
            if '_frames' not in k:
                raise AssertionError("All vector metrics should have the '_frames' suffix (found: '{}').".format(k))
            # A regression is computed on each time frame (considered independent)
            for frame_i in range(metric.shape[2]):
                results_list.append(_compute_metric_statistics(metric[:, :, frame_i], k + '{:02d}'.format(frame_i)))
        elif len(metric.shape) == 4:  # 2D metric (matrix for each step of each sequence) e.g. MFCCs, spectral contrast
            continue  # TODO multivariate correlation coeff ?
            # MFCC: orthogonal features (in the frequency axis)
        else:
            raise NotImplementedError()
    # Compute global averages (_frames excluded)
    results_df = pd.DataFrame(results_list)
    df_without_frames = results_df[~results_df['metric'].str.contains('_frame')]
    global_averages = {'metric': 'global_average'}
    for k in list(df_without_frames.columns):
        if k != 'metric':
            global_averages[k] = df_without_frames[k].values.mean()
    results_df.loc[len(results_df)] = global_averages  # loc can add a new row to a DataFrame
    # TODO output results and/or store
    return results_df


if __name__ == "__main__":
    _storage_path = "/home/gwendal/Jupyter/nn-synth-interp/saved/" \
                    "FlowReg_dimz5020/CElabels_smooth0.2_noise0.1__permsFalse/interp_validation"
    #_storage_path = '/media/gwendal/Data/Interpolations/LinearNaive/interp_validation'
    #_storage_path = '/media/gwendal/Data/Interpolations/ThresholdNaive/interp_validation'
    _results_df = process_features(pathlib.Path(_storage_path))


