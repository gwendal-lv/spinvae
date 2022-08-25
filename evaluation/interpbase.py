"""
Base class to compute metrics about an abstract interpolation method
"""
import json
import os.path
import pathlib
import pickle
import shutil
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
import soundfile
import torch
import matplotlib.pyplot as plt
import librosa

import data
import data.preset
from data.abstractbasedataset import PresetDataset
import evaluation.load
import utils.figures


class InterpSequence:
    def __init__(self, parent_path: pathlib.Path, seq_index: int):
        """ Class for loading/storing an interpolation sequence (audio and/or spectrograms output).

        :param parent_path: Path to the folder where all sequences are to be stored (each in its own folder).
        :param seq_index: Index (or ID) of that particular sequence; its own folder will be named after this index.
        """
        self.parent_path = parent_path
        self.seq_index = seq_index
        self.name = ''

        self.u = np.asarray([], dtype=float)
        self.UID_start, self.UID_end = -1, -1
        self.audio = list()
        self.spectrograms = list()
        self.interpolation_metrics = dict()
        # Actual number of frames might be greater (odd) because of window centering and signal padding
        self.num_metric_frames = 8  # 500ms frames for 4.0s audio  FIXME use ctor arg

    @property
    def storage_path(self) -> pathlib.Path:
        return self.parent_path.joinpath('{:05d}'.format(self.seq_index))

    def process_and_save(self):
        """ Computes interpolation metrics
            then saves all available data into a new directory created inside the parent dir. """
        self.compute_interpolation_metrics()
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=False)
        with open(self.storage_path.joinpath('info.json'), 'w') as f:
            json.dump({'sequence_index': self.seq_index, 'sequence_name': self.name,
                       'num_steps': len(self.audio), 'start': int(self.UID_start), 'end': int(self.UID_end)}, f)
        for step_i, audio_Fs in enumerate(self.audio):
            soundfile.write(self.storage_path.joinpath('audio_step{:02d}.wav'.format(step_i)),
                            audio_Fs[0], audio_Fs[1], subtype='FLOAT')
        with open(self.storage_path.joinpath('spectrograms.pkl'), 'wb') as f:
            pickle.dump(self.spectrograms, f)
        with open(self.storage_path.joinpath('interp_metrics.dict.pkl'), 'wb') as f:
            pickle.dump(self.interpolation_metrics, f)
        # TODO
        #     title
        #     plot spectral characteristics (3rd of 4th line)
        fig, axes = utils.figures.plot_spectrograms_interp(
            self.u, torch.vstack([torch.unsqueeze(torch.unsqueeze(s, dim=0), dim=0) for s in self.spectrograms]),
            metrics=self.interpolation_metrics, plot_delta_spectrograms=False,
            title=(self.name if len(self.name) > 0 else None)
        )
        fig.savefig(self.storage_path.joinpath("spectrograms_interp.pdf"))
        plt.close(fig)
        fig.savefig(self.storage_path.joinpath("spectrograms_interp.png"))
        plt.close(fig)

    def compute_interpolation_metrics(self):
        """ Compute the internal dict of metrics.
            Metrics keys ending with '_frames' are 2D metrics ; other metrics are average values across all frames.
            First dim of all arrays is always the interpolation step.
        """
        # FIXME use matlab's TimbreToolbox instead
        self.interpolation_metrics = dict()
        # first, compute librosa spectrograms
        sr = self.audio[0][1]
        n_fft = 2048
        n_samples = self.audio[0][0].shape[0]
        frame_len = n_samples // self.num_metric_frames
        specs = [np.abs(librosa.stft(a[0], n_fft=n_fft, hop_length=n_fft//2)) for a in self.audio]  # Linear specs
        # metrics "mask" if RMS volume is lower than a threshold  TODO use fft hop length
        rms = np.asarray([librosa.feature.rms(y=a[0], frame_length=n_fft, hop_length=n_fft//2)[0] for a in self.audio])
        self.interpolation_metrics['rms'] = 10.0 * np.log10(np.mean(rms, axis=1))
        rms_mask = 10.0 * np.log10(rms)  # same number of frames for all other features
        rms = 10.0 * np.log10(self._reduce_num_frames(rms, num_frames=2 * self.num_metric_frames))  # More RMS frames
        self.interpolation_metrics['rms_frames'] = rms
        # harmonic/percussive/residuals separation ; get average value for each (1 frame for each interp step)
        h_p_separated = [librosa.decompose.hpss(s, margin=(1.0, 3.0)) for s in specs]
        res = [np.abs(specs[i] - (hp[0]+hp[1])) for i, hp in enumerate(h_p_separated)]
        harm = np.asarray([20.0 * np.log10(hp[0] + 1e-7).mean() for hp in h_p_separated])
        perc = np.asarray([20.0 * np.log10(hp[1] + 1e-7).mean() for hp in h_p_separated])
        res = np.asarray([20.0 * np.log10(r + 1e-7).mean() for r in res])
        # hpss: Absolute values are much less interesting than ratios (log diffs)
        self.interpolation_metrics['harm_perc_diff'] = harm - perc
        self.interpolation_metrics['harm_residu_diff'] = harm - res
        # Chromagrams (will use their own Constant-Q Transform)
        chromagrams = [librosa.feature.chroma_cqt(y=a[0], sr=sr, hop_length=n_fft//2) for a in self.audio]
        """
        chromagrams = [c / c.sum(axis=0)[np.newaxis, :] for c in chromagrams]  # convert to probabilities
        chroma_value_tiles = np.tile(np.arange(12), (rms_mask.shape[1], 1)).T
        chroma_values = [c * chroma_value_tiles for c in chromagrams]  # Weight of each chroma * its 'scalar' value
        avg_chroma_value_frames = [np.sum(c_v, axis=0) for c_v in chroma_values]
        avg_chroma_values = [avg_c[rms_mask[i, :] > -120.0].mean() for i, avg_c in enumerate(avg_chroma_value_frames)]
        self.interpolation_metrics['chroma_value'] = np.asarray(avg_chroma_values)
        # Compute the std for each pitch (across all 'non-zero-volume' frames) then average all stds
        chroma_std = [np.std(c[:, rms_mask[i]>-120.0], axis=1).mean() for i, c in enumerate(chromagrams)]
        self.interpolation_metrics['chroma_std'] = np.asarray(chroma_std)
        """
        # Getting the argmax and the corresponding std is probably a simpler/better usage of chromas
        chroma_argmax = [np.argmax(c, axis=0)[rms_mask[i]>-120.0] for i, c in enumerate(chromagrams)]
        self.interpolation_metrics['chroma_argmax_avg'] = np.asarray([c.mean() for c in chroma_argmax])
        self.interpolation_metrics['chroma_argmax_std'] = np.asarray([c.std() for c in chroma_argmax])
        # 1D spectral features: centroid, rolloff, etc... (use mask to compute average)
        spec_features = np.log(np.asarray([librosa.feature.spectral_centroid(S=s, sr=sr)[0] for s in specs]))
        self._mask_rms_and_add_1D_spectral_metric(spec_features, 'spec_centroid', rms_mask)
        spec_features = 20.0 * np.log10(np.asarray([librosa.feature.spectral_flatness(S=s)[0] for s in specs]))
        self._mask_rms_and_add_1D_spectral_metric(spec_features, 'spec_flatness')  # no mask for flatness
        spec_features = np.log(np.asarray([librosa.feature.spectral_rolloff(S=s, sr=sr)[0] for s in specs]))
        self._mask_rms_and_add_1D_spectral_metric(spec_features, 'spec_rolloff')  # no mask for roll-off either
        spec_features = np.log(np.asarray([librosa.feature.spectral_bandwidth(S=s, sr=sr)[0] for s in specs]))
        self._mask_rms_and_add_1D_spectral_metric(spec_features, 'spec_bandwidth')  # no mask for bandwidth
        # TODO spectral contrast and MFCCs: 2D features - will be processed later
        spec_features = np.asarray([librosa.feature.mfcc(y=a[0], sr=sr, n_mfcc=13, hop_length=n_fft//2)
                                    for a in self.audio])
        self.interpolation_metrics['mfcc_full'] = spec_features  # will be processed later
        spec_features = np.asarray([librosa.feature.spectral_contrast(S=s, sr=sr) for s in specs])
        self.interpolation_metrics['spec_contrast_full'] = spec_features  # will be processed later
        # use https://github.com/AudioCommons/timbral_models? No peer-review of this, not much information...
        # Better: Timbral Toolbox (MATLAB, G. Peeters 2011, updated)

    def _reduce_num_frames(self, m: np.ndarray, num_frames: Optional[int] = None):
        """ Reduces (using averaging) the number of frames of a given metric (1D or 2D for a given interpolation step).

        :param m: metric provided as a 2D or 3D numpy array.
        """
        # First output frame will use a smaller number of input frames (usually short attack)
        if len(m.shape) == 1:
            raise AssertionError("Please provide a 2D or 3D metric (must contain several frames for each interp step)")
        if num_frames is None:
            num_frames = self.num_metric_frames
        avg_len = 1 + m.shape[1] // num_frames
        first_len = m.shape[1] - (num_frames-1) * avg_len
        m_out = np.zeros((m.shape[0], num_frames))
        if len(m.shape) == 2:
            m_out[:, 0] = np.mean(m[:, 0:first_len], axis=1)
            for i in range(1, num_frames):
                i_in = first_len + (i-1) * avg_len
                m_out[:, i] = np.mean(m[:, i_in:i_in+avg_len], axis=1)
        else:
            raise NotImplementedError()
        return m_out

    def _mask_rms_and_add_1D_spectral_metric(self, m: np.ndarray, name: str, rms: Optional[np.ndarray] = None):
        valid_m = [m[i, :][rms[i, :] > -120.0] for i in range(rms.shape[0])] if rms is not None else m
        self.interpolation_metrics[name + '_avg'] = np.asarray([_m.mean() for _m in valid_m])
        self.interpolation_metrics[name + '_std'] = np.asarray([_m.std() for _m in valid_m])

    def load(self):
        """ Loads a sequence using previously rendered data. """
        with open(self.storage_path.joinpath('info.json'), 'r') as f:
            json_info = json.load(f)
            self.UID_start, self.UID_end, self.name, num_steps = \
                json_info['start'], json_info['end'], json_info['sequence_name'], json_info['num_steps']
        with open(self.storage_path.joinpath('interp_metrics.dict.pkl'), 'rb') as f:
            self.interpolation_metrics = pickle.load(f)
        with open(self.storage_path.joinpath('spectrograms.pkl'), 'rb') as f:
            self.spectrograms = pickle.load(f)
        self.audio = list()
        for step_i in range(num_steps):
            self.audio.append(soundfile.read(self.storage_path.joinpath('audio_step{:02d}.wav'.format(step_i))))

    # TODO render independent spectrograms to PNG (for future website)


class LatentInterpSequence(InterpSequence):
    def __init__(self, parent_path: pathlib.Path, seq_index: int):
        super().__init__(parent_path=parent_path, seq_index=seq_index)
        self.z = torch.empty((0, ))


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
        self.all_interp_metrics = dict()

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

    def interp_metrics_init(self):
        self.all_interp_metrics = None

    def append_interp_metric(self, m: InterpSequence):
        """ Internally stores all available metrics from the given sequence. """
        if self.all_interp_metrics is None:  # Create dict and init lists of arrays
            self.all_interp_metrics = {k: list() for k in m.interpolation_metrics}
        for k, v in m.interpolation_metrics.items():
            self.all_interp_metrics[k].append(v)

    def store_interp_metrics(self):
        stacked_metrics = {k: np.stack(l, axis=0) for k, l in self.all_interp_metrics.items()}
        with open(self.storage_path.joinpath('all_interp_metrics.pkl'), 'wb') as f:
            pickle.dump(stacked_metrics, f)


class NaivePresetInterpolation(InterpBase):
    def __init__(self, dataset, dataset_type, dataloader, base_storage_path: str,
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

        self.interp_metrics_init()

        current_sequence_index = 0
        # Retrieve all latent vectors that will be used for interpolation
        # We assume that batch size is an even number...
        for batch_idx, sample in enumerate(self.dataloader):
            end_sequence_with_next_item = False  # If True, the next item is the 2nd (last) of an InterpSequence
            x_in, v_target, sample_info = sample[0], sample[1], sample[2]
            for i in range(sample[0].shape[0]):
                if self.verbose:
                    print("Processing item {}".format(current_sequence_index*2 + (i%2)))
                if not end_sequence_with_next_item:
                    end_sequence_with_next_item = True
                else:
                    # It's easier to compute interpolations one-by-one (all data might not fit into RAM)
                    seq = InterpSequence(self.storage_path, current_sequence_index)
                    seq.UID_start, seq.UID_end = sample_info[i-1, 0], sample_info[i, 0]
                    seq.name = self.get_sequence_name(seq.UID_start, seq.UID_end, self.dataset)
                    seq.u = self.get_u_interpolated()
                    # Convert learnable presets to VST presets  FIXME works for Dexed only
                    start_end_presets = data.preset.DexedPresetsParams(
                        self.dataset, learnable_presets=v_target[i-1:i+1, :])
                    start_end_presets = start_end_presets.get_full().clone().cpu().numpy()
                    vst_presets = self.get_interpolated_presets(seq.u, start_end_presets)

                    seq.audio, seq.spectrograms = self.generate_audio_and_spectrograms(vst_presets)
                    seq.process_and_save()
                    self.append_interp_metric(seq)

                    current_sequence_index += 1
                    end_sequence_with_next_item = False
            if batch_idx >= 1:  # FIXME, TEMP
                break
        self.store_interp_metrics()
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
        spectrograms = [self.dataset.compute_spectrogram(audio_renders[i][0])
                        for i in range(len(audio_renders))]
        return audio_renders, spectrograms


class ModelBasedInterpolation(InterpBase):
    def __init__(self, model_loader: Optional[evaluation.load.ModelLoader] = None,
                 device='cpu', num_steps=7,
                 u_curve='arcsin', latent_interp_kind='linear', verbose=True):
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

        self.interp_metrics_init()
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
                    self.append_interp_metric(seq)

                    current_sequence_index += 1
                    end_sequence_with_next_item = False

            # FIXME TEMP
            if batch_idx >= 1:
                break

        self.store_interp_metrics()

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
    :returns: (sum_squared_normalized_residuals, linregress_R2, pearson_r_squared
    """
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


