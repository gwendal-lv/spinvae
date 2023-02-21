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
from evalconfig import InterpEvalConfig
from evaluation.interpsequence import InterpSequence, LatentInterpSequence
from utils.timbretoolbox import InterpolationTimbreToolbox
import utils.stat
import utils.math


class InterpBase(ABC):
    def __init__(self, num_steps=7, u_curve='linear', verbose=True,
                 reference_storage_path: Optional[pathlib.Path] = None,
                 verbose_postproc=True,
                 **kwargs):
        """
        Base attributes and methods of any interpolation engine.

        :param num_steps:
        :param u_curve: The type of curve used for the interpolation abscissa u.
        """
        self.num_steps = num_steps
        self.u_curve = u_curve
        self.verbose = verbose
        self.verbose_postproc = verbose_postproc
        self._reference_storage_path = reference_storage_path
        # keyword arguments
        self.use_reduced_dataset = False  # faster debugging
        self.generated_note_duration = None  # None corresponds to dataset's default note duration. For interp gen only
        for k, v in kwargs.items():
            if k == 'use_reduced_dataset':
                assert isinstance(v, bool)
                self.use_reduced_dataset = v
            elif k == 'generated_note_duration':
                assert v is None or (isinstance(v, tuple) and len(v) == 2)
                self.generated_note_duration = v
            else:
                raise KeyError(f"Unexpected kwarg name '{k}'")

    def get_u_interpolated(self, extrapolate_left=0, extrapolate_right=0):
        """ Returns the interpolation 'time steps', usually in [0.0, 1.0];
        may be < 0.0 and/or > 1.0 if extrapolation is required

        :param extrapolate_left: Number of extra time steps < 0.0
        :param extrapolate_right: Number of extra time steps > 1.0
        """
        linear_u = np.linspace(
            0 - extrapolate_left, self.num_steps - 1 + extrapolate_right,
            self.num_steps + extrapolate_left + extrapolate_right, endpoint=True
        ) / (self.num_steps - 1)
        if self.u_curve == 'linear':
            return linear_u
        elif self.u_curve == 'arcsin':
            assert extrapolate_left == 0 and extrapolate_right == 0, "arcsin u_curve shouldn't be used to extrapolate"
            return 0.5 + np.arcsin(np.linspace(-1.0, 1.0, self.num_steps, endpoint=True)) / np.pi
        elif self.u_curve == 'threshold':
            return (linear_u > 0.5).astype(float)
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
    def contains_eval_data(storage_path):
        return os.path.exists(storage_path)

    @staticmethod
    def get_sequence_name(start_UID: int, end_UID: int, dataset: PresetDataset):
        start_name = dataset.get_name_from_preset_UID(start_UID)
        end_name = dataset.get_name_from_preset_UID(end_UID)
        return "[{}] '{}' --> [{}] '{}'".format(start_UID, start_name, end_UID, end_name)

    def try_process_dataset(self, force_re_eval=False, skip_audio_render=False):
        """ Returns True if the evaluation is actually performed, False otherwise. """
        if force_re_eval:
            self.process_dataset(skip_audio_render)
            return True
        else:
            if os.path.exists(self.storage_path):
                print("[{}] Some results were already stored in '{}' - dataset won't be re-evaluated"
                      .format(type(self).__name__, self.storage_path))
                return False
            else:
                self.process_dataset(skip_audio_render)
                return True

    def process_dataset(self, skip_audio_render=False):
        if not skip_audio_render:
            self.render_audio()
        else:
            if self.verbose:
                print("[{}] Skip audio render - will compute interpolation metrics only".format(type(self).__name__))
        self.compute_and_save_interpolation_metrics()

    def on_render_audio_begins(self):
        self.create_storage_directory()
        self._audio_render_t_start = datetime.now()

    @abstractmethod
    def render_audio(self):
        pass

    def on_render_audio_minibatch_ends(self, batch_idx: int, num_batches: int):
        """
        :return: forced_stop (usually False, may be True during debug to evaluate quickly)
        """
        if self.verbose:
            print("[{}] Minibatch {}/{} rendered to audio files ({:.0f} %)"
                  .format(type(self).__name__, batch_idx+1, num_batches, 100.0 * (batch_idx+1) / num_batches))
        if self.use_reduced_dataset and batch_idx >= 0:  # during debug: process a few mini-batch(es) only
            warnings.warn("self.use_reduced_dataset is True; finished rendering audio files")
            return True
        return False

    def on_render_audio_ends(self, num_interp_sequences: int):
        with open(self.storage_path.joinpath('audio_renders_info.json'), 'w') as f:
            json.dump(
                {'num_sequences': num_interp_sequences, 'num_steps_per_sequence': self.num_steps,
                 'custom_note_duration': self.generated_note_duration},
                f
            )
        if self.verbose:
            delta_t = (datetime.now() - self._audio_render_t_start).total_seconds()
            print("[{}] Finished rendering audio for {} interpolations in {:.1f}min total ({:.1f}s / interpolation)"
                  .format(type(self).__name__, num_interp_sequences, delta_t / 60.0, delta_t / num_interp_sequences))

    def get_stored_postproc_sequences_descriptors(self) -> List[pd.DataFrame]:
        """ Returns the dataframes of sequences descriptors (audio features) previously computed for this model. """
        return InterpolationTimbreToolbox.get_stored_postproc_sequences_descriptors(self.storage_path)

    def get_stored_reference_postproc_sequences_descriptors(self) -> List[pd.DataFrame]:
        """ Returns the dataframes of sequences descriptors (audio features) obtained from the reference model
         (may be this model itself). """
        if self._reference_storage_path is None:
            return self.get_stored_postproc_sequences_descriptors()
        else:
            return InterpolationTimbreToolbox.get_stored_postproc_sequences_descriptors(self._reference_storage_path)

    def compute_and_save_interpolation_metrics(self):
        """
        Compute features for each individual audio file (which has already been rendered),
        then compute interpolation metrics for each sequence.
        """
        self.compute_store_timbre_toolbox_features()

        all_seqs_dfs = self.get_stored_postproc_sequences_descriptors()
        all_reference_seqs_dfs = self.get_stored_reference_postproc_sequences_descriptors()
        features_stats = InterpolationTimbreToolbox.get_default_postproc_features_stats()
        interp_results = self._compute_interp_metrics(all_seqs_dfs, features_stats, all_reference_seqs_dfs)
        with open(self.storage_path.joinpath('interp_results.pkl'), 'wb') as f:
            pickle.dump(interp_results, f)

    def compute_store_timbre_toolbox_features(self):
        _timbre_toolbox_path = '~/Documents/MATLAB/timbretoolbox'
        timbre_proc = InterpolationTimbreToolbox(
            _timbre_toolbox_path, self.storage_path, num_matlab_proc=8, remove_matlab_csv_after_usage=True,
            verbose=self.verbose_postproc
        )
        timbre_proc.run()
        timbre_proc.post_process_features(self.storage_path)

    @staticmethod
    def get_interp_results(storage_path: pathlib.Path, eval_config: Optional[InterpEvalConfig] = None):
        with open(storage_path.joinpath('interp_results.pkl'), 'rb') as f:
            multi_metrics_interp_results = pickle.load(f)
        if eval_config is not None:
            # filtering: remove unused columns (e.g. min / max), ...
            for metric_name, interp_results_df in multi_metrics_interp_results.items():
                for col in list(interp_results_df.columns):
                    try:
                        if eval_config.exclude_min_max_interp_features:
                            if col.endswith('_min') or col.endswith('_max'):
                                interp_results_df.drop(columns=[col], inplace=True)  # must be inplace inside the for loop
                        for feat_name in eval_config.excluded_interp_features:
                            if col.startswith(feat_name):
                                interp_results_df.drop(columns=[col], inplace=True)  # must be inplace inside the for loop
                    except KeyError:
                        pass  # A feature might be removed multiple times (multiple overlapping removal criteria)
        return multi_metrics_interp_results

    @staticmethod
    def _compute_interp_metrics(all_seqs_dfs: List[pd.DataFrame],
                                features_stats: Dict[str, Any],
                                all_reference_seqs_dfs: List[pd.DataFrame]):
        """
        Quantifies how smooth and linear the interpolation is, using previously computed audio features
        from timbretoolbox TODO maybe include some librosa features

        :param features_stats: Must have been computed once, and the same must be used for all sequences
        """
        # Compute interpolation performance for each interpolation metric, for each sequence
        interp_results = {'smoothness': list(), 'nonlinearity': list()}
        for seq_idx, seq_df in enumerate(all_seqs_dfs):
            ref_seq_df = all_reference_seqs_dfs[seq_idx]
            seq_interp_results = InterpBase._compute_sequence_interp_metrics(seq_df, features_stats, ref_seq_df)
            for k in seq_interp_results:
                interp_results[k].append(seq_interp_results[k])
        # Then sum each interp metric up into a single df
        for k in interp_results:
            interp_results[k] = pd.DataFrame(interp_results[k])
        return interp_results

    @staticmethod
    def _compute_sequence_interp_metrics(seq: pd.DataFrame, features_stats: Dict[str, Any], ref_seq: pd.DataFrame):
        interp_metrics = {'smoothness': dict(), 'nonlinearity': dict()}
        seq = seq.drop(columns="step_index")
        step_h = 1.0 / (len(seq) - 1.0)
        for col in seq.columns:
            feature_values = ((seq[col] - features_stats['mean'][col]) / features_stats['std'][col]).values
            # Smoothness: https://proceedings.neurips.cc/paper/2019/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf
            # Second-order central difference using a conv kernel, then compute the RMS of the smaller array
            smoothness = np.convolve(feature_values, [1.0, -2.0, 1.0], mode='valid') / (step_h ** 2)
            interp_metrics['smoothness'][col] = np.sqrt( (smoothness ** 2).mean() )
            # non-linearity, quantified as the RMS of the error vs. the ideal linear curve
            #    by using a reference, we ensure that we always use the exact same start/end points
            ref_feature_values = ((ref_seq[col] - features_stats['mean'][col]) / features_stats['std'][col]).values
            target_linear_values = np.linspace(ref_feature_values[0], ref_feature_values[-1],
                                               num=ref_feature_values.shape[0]).T
            interp_metrics['nonlinearity'][col] = np.sqrt( ((feature_values - target_linear_values) ** 2).mean() )
        return interp_metrics

    @staticmethod
    def compute_interp_improvement_vs_ref(eval_config: InterpEvalConfig):
        ref_interp_results = InterpBase.get_interp_results(eval_config.ref_model_interp_path, eval_config)
        models_interp_results = [
            InterpBase.get_interp_results(m_config['interp_storage_path'], eval_config)
            for m_config in eval_config.other_models
        ]
        # We'll build a wide-form DF from a list of dicts
        improvements_df = []
        for model_idx, interp_results in enumerate(models_interp_results):
            interp_config = eval_config.other_models[model_idx]
            # retrieve model hparams (train/model and interpolation hparams)
            nn_model_config, nn_train_config = evaluation.load.ModelLoader.get_model_train_configs(
                interp_config['base_model_path'])
            nn_model_config_dict = {'mdlcfg__' + k: v for k, v in nn_model_config.__dict__.items()}
            nn_train_config_dict = {'trncfg__' + k: v for k, v in nn_train_config.__dict__.items()}
            # also 'manually' add interp hparams
            interp_config_dict = {'u_curve': interp_config['u_curve'], 'z_curve': interp_config['latent_interp'],
                                  'refine_level': interp_config['refine_level']}
            # Then process all interpolation metrics (e.g. smoothness, ....) for all audio features
            for metric_name in ref_interp_results.keys():
                ref_interp_df = ref_interp_results[metric_name]
                model_interp_df = interp_results[metric_name]
                # Average variation (vs. reference) of the median and mean of features
                #    Average is not weighted: all features are considered to be as equally important
                median_variation_vs_ref = (model_interp_df.median() - ref_interp_df.median()) / ref_interp_df.median()
                median_variation_vs_ref = median_variation_vs_ref.values.mean()
                mean_variation_vs_ref = (model_interp_df.mean() - ref_interp_df.mean()) / ref_interp_df.mean()
                mean_variation_vs_ref = mean_variation_vs_ref.values.mean()
                # Wilcoxon test: which medians have significantly improved? (have been reduced significantly?)
                improved_test_results = utils.stat.wilcoxon_test(ref_interp_df, model_interp_df)
                # Other test: which medians are now significantly higher?
                deteriorated_test_results = utils.stat.wilcoxon_test(model_interp_df, ref_interp_df)
                # Add everything to the dataframe
                improvements_df.append({
                    'model': interp_config['model_interp_name'], 'metric': metric_name,
                    'wilcoxon_improved_features': np.count_nonzero(improved_test_results[1].values),
                    'wilcoxon_deteriorated_features': np.count_nonzero(deteriorated_test_results[1].values),
                    'median_variation_vs_ref': median_variation_vs_ref,
                    'mean_variation_vs_ref': mean_variation_vs_ref,
                    **nn_model_config_dict, **nn_train_config_dict, **interp_config_dict
                })
        improvements_df = pd.DataFrame(improvements_df)
        return improvements_df


class NaivePresetInterpolation(InterpBase):
    def __init__(self, dataset, dataset_type, dataloader, storage_path: Union[str, pathlib.Path],
                 num_steps=7, u_curve='linear', verbose=True, verbose_postproc=True,
                 reference_storage_path: Optional[pathlib.Path] = None,
                 **kwargs):
        super().__init__(num_steps, u_curve, verbose, reference_storage_path, verbose_postproc=verbose_postproc,
                         **kwargs)
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.dataloader = dataloader
        self._storage_path = pathlib.Path(storage_path)

    @property
    def storage_path(self) -> pathlib.Path:
        return self._storage_path

    def render_audio(self):
        """ Generates interpolated sounds using the 'naïve' linear interpolation between VST preset parameters. """
        self.on_render_audio_begins()  # Creates the storage directory

        current_sequence_index = 0
        # Retrieve all latent vectors that will be used for interpolation
        # We assume that batch size is an even number...
        for batch_idx, minibatch in enumerate(self.dataloader):
            end_sequence_with_next_item = False  # If True, the next item is the 2nd (last) of an InterpSequence
            v_in, uid = minibatch[1], minibatch[2]
            for i in range(v_in.shape[0]):
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
            forced_stop = self.on_render_audio_minibatch_ends(batch_idx, len(self.dataloader))
            if forced_stop:
                break
        self.on_render_audio_ends(current_sequence_index)

    def get_interpolated_presets(self, u: np.ndarray, start_end_presets: np.ndarray):
        interp_f = scipy.interpolate.interp1d(
            [0.0, 1.0], start_end_presets, kind='linear', axis=0,
            bounds_error=True)  # extrapolation disabled, no fill_value
        return interp_f(u)

    def generate_audio_and_spectrograms(self, vst_presets: np.ndarray):
        midi_pitch, midi_vel = self.dataset.default_midi_note
        # FIXME custom note length
        audio_renders = [
            self.dataset._render_audio(vst_presets[i, :], midi_pitch, midi_vel, self.generated_note_duration)
            for i in range(vst_presets.shape[0])
        ]
        spectrograms = [self.dataset.compute_spectrogram(a[0]) for a in audio_renders]
        return audio_renders, spectrograms


class ModelBasedInterpolation(InterpBase):
    def __init__(
            self, model_loader: Optional[evaluation.load.ModelLoader] = None, device='cpu', num_steps=7,
            u_curve='linear', latent_interp_kind='linear', verbose=True,
            storage_path: Optional[pathlib.Path] = None, reference_storage_path: Optional[pathlib.Path] = None,
            verbose_postproc=True,
            **kwargs
    ):
        """
        A class for performing interpolations using a neural network model whose inputs are latent vectors.

        :param model_loader: If given, most of other arguments (related to the model and corresponding
         dataset) will be ignored.
        """
        super().__init__(num_steps=num_steps, verbose=verbose, u_curve=u_curve,
                         reference_storage_path=reference_storage_path, verbose_postproc=verbose_postproc,
                         **kwargs)
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

    def get_stored_reference_postproc_sequences_descriptors(self) -> List[pd.DataFrame]:
        if self._reference_storage_path is None:
            raise AssertionError("A reference interpolation-model must be provided.")
        else:
            return super().get_stored_reference_postproc_sequences_descriptors()

    def render_audio(self):
        """ Performs an interpolation over the whole given dataset (usually validation or test), using pairs
        of items from the dataloader. Dataloader should be deterministic. Total number of interpolations computed:
        len(dataloder) // 2. """
        self.on_render_audio_begins()
        self.ae_model = self.ae_model.to(self.device)

        # to store all latent-specific stats (each sequence is written to SSD before computing the next one)
        # encoded latent vectors (usually different from endpoints, if the corresponding preset is not 100% accurate)
        z_ae = list()
        # Interpolation endpoints
        z_endpoints = list()

        # Accuracy / L1 error improvement, if the latent code is refined
        z_refinement_results = {
            'acc_first_guess': [], 'acc_refined': [], 'l1_err_first_guess': [], 'l1_err_refined': []}
        def _append_refinement_results(acc, L1_err, acc_1st, L1_err_1st):
            z_refinement_results['acc_first_guess'].append(acc_1st)
            z_refinement_results['acc_refined'].append(acc)
            z_refinement_results['l1_err_first_guess'].append(L1_err_1st)
            z_refinement_results['l1_err_refined'].append(L1_err)

        current_sequence_index = 0
        # Retrieve all latent vectors that will be used for interpolation
        # We assume that batch size is an even number...
        for batch_idx, minibatch in enumerate(self.dataloader):
            end_sequence_with_next_item = False  # If True, the next item is the 2nd (last) of an InterpSequence
            x_in, v_in, uid, notes, label = [m.to(self.device) for m in minibatch]
            N = x_in.shape[0]
            for i in range(N):
                if not end_sequence_with_next_item:
                    end_sequence_with_next_item = True
                else:
                    # It's easier to compute interpolations one-by-one (all data might not fit into RAM)
                    seq = LatentInterpSequence(
                        self.storage_path, current_sequence_index, uid[i-1].item(), uid[i].item(),
                        self.get_sequence_name(uid[i-1].item(), uid[i].item(), self.dataset)
                    )
                    z_start, z_start_first_guess, acc, L1_err, acc_1st, L1_err_1st = self.compute_latent_vector(
                        x_in[i-1:i], v_in[i-1:i], uid[i-1:i], notes[i-1:i])
                    _append_refinement_results(acc, L1_err, acc_1st, L1_err_1st)
                    z_end, z_end_first_guess, acc, L1_err, acc_1st, L1_err_1st = self.compute_latent_vector(
                        x_in[i:i+1], v_in[i:i+1], uid[i:i+1], notes[i:i+1])
                    _append_refinement_results(acc, L1_err, acc_1st, L1_err_1st)
                    z_ae.append(z_start_first_guess), z_ae.append(z_end_first_guess)
                    z_endpoints.append(z_start), z_endpoints.append(z_end)

                    seq.u, seq.z = self.interpolate_latent(z_start, z_end)
                    seq.audio, seq.spectrograms = self.generate_audio_and_spectrograms(seq.z)
                    seq.save()

                    current_sequence_index += 1
                    end_sequence_with_next_item = False
            forced_stop = self.on_render_audio_minibatch_ends(batch_idx, len(self.dataloader))
            if forced_stop:
                break

        z_refinement_results['z_first_guess'] = torch.vstack(z_ae).detach().clone().cpu().numpy()
        all_z_endpoints = torch.vstack(z_endpoints).detach().clone().cpu().numpy()
        z_refinement_results['z_estimated'] = all_z_endpoints
        with open(self.storage_path.joinpath('z_refinement_results.pkl'), 'wb') as f:
            pickle.dump(z_refinement_results, f)
        self.on_render_audio_ends(current_sequence_index)

    @abstractmethod
    def compute_latent_vector(self, x_in, v_in, uid, notes) \
            -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
        """ Computes the most appropriate latent vector (child class implements this method)

        :returns: z_estimated, z_first_guess,
            preset_accuracy, preset_L1_error, preset_accuracy_1st_guess, preset_L1_error_1st_guess
        """
        pass

    def interpolate_latent(
            self, z_start, z_end, extrapolate_left=0, extrapolate_right=0
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """ Returns a N x D tensor of interpolated latent vectors, where N is the number of interpolation steps (here:
        considered as a batch size) and D is the latent dimension. Each latent coordinate is interpolated independently.

        Non-differentiable: based on scipy.interpolate.interp1d.

        :param z_start: 1 x D tensor
        :param z_end: 1 x D tensor
        :param extrapolate_left: Number of steps to extrapolate for u < 0
        :param extrapolate_right: Number of steps to extrapolate for u > 0
        :returns: u, interpolated_z
        """
        assert extrapolate_left >= 0 and extrapolate_right >= 0
        extrapolate = (extrapolate_left > 0 or extrapolate_right > 0)
        u_interpolated = self.get_u_interpolated(extrapolate_left, extrapolate_right)

        z_start_end = torch.cat([z_start, z_end], dim=0).clone().detach().cpu().numpy()
        if self.latent_interp_kind == 'linear':
            interp_f = scipy.interpolate.interp1d(
                [0.0, 1.0], z_start_end, kind='linear', axis=0,
                bounds_error=(not extrapolate), fill_value=("extrapolate" if extrapolate else np.nan)
            )
            z_interpolated = interp_f(u_interpolated)
        elif self.latent_interp_kind == 'spherical':
            sph_interpolator = utils.math.SphericalInterpolator(z_start_end[0, :], z_start_end[1, :])
            z_interpolated = sph_interpolator(u_interpolated)
        else:
            raise AssertionError("latent interpolation '{}' not available".format(self.latent_interp_kind))
        # FIXME remove this debug check
        assert np.allclose(z_start_end[0, :], z_interpolated[extrapolate_left, :], atol=1e-6), "(Numerical?) error for start z"
        assert np.allclose(z_start_end[1, :], z_interpolated[extrapolate_left + self.num_steps - 1, :], atol=1e-6), "(Numerical?) error for end z"
        return u_interpolated, torch.tensor(z_interpolated, device=self.device, dtype=torch.float32)

    @abstractmethod
    def generate_audio_and_spectrograms(self, z: torch.Tensor):
        """ Returns a list of audio waveforms and/or a list of spectrogram corresponding to latent vectors z (given
            as a 2D mini-batch of vectors). """
        pass

