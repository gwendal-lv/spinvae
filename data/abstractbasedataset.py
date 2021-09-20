""" TODO proper doc """

import os
import pathlib
from abc import ABC, abstractmethod  # Abstract Base Class
from typing import Sequence, Optional

import pandas as pd
import json
from datetime import datetime
import multiprocessing

import torch
import torch.utils
import numpy as np

import utils.audio

from data.preset import PresetsParams, PresetIndexesHelper

# See https://github.com/pytorch/audio/issues/903
#torchaudio.set_audio_backend("sox_io")


class AudioDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, note_duration,
                 n_fft, fft_hop, Fs,  # ftt 1024 hop=512: spectrogram is approx. the size of 5.0s@22.05kHz audio
                 midi_notes=((60, 100),),
                 multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=30.0, mel_fmax=11e3,
                 normalize_audio=False, spectrogram_min_dB=-120.0,
                 spectrogram_normalization: Optional[str] = 'min_max',
                 data_storage_path: Optional[str] = None,
                 random_seed=0, data_augmentation=True):
        """
        Abstract Base Class for any dataset of audio samples (from a synth of from an acoustic instrument).
        A preset UID corresponds to a unique synth preset or acoustic instrument recording.

        This abstract class provides itself a few functionalities:
            - single or stacked spectrograms
            - spectrograms generation from wav files
            - computes statistics on spectrogram, for normalization

        It can be inherited by a concrete dataset class such as :
            - a fixed dataset of audio samples (e.g. NSynth)
            - a generated dataset of audio samples (e.g. Surge)
            - a generated dataset of audio samples + associated presets (e.g. Dexed)

        :param note_duration: Tuple: MIDI Note (on_duration, off_duration) in seconds
        :param n_fft: Width of the FFT window for spectrogram computation
        :param fft_hop: STFT hop length (in samples)
        :param midi_notes: Tuple of (midi_pitch, midi_velocity) tuples of notes that should be rendered. Length
            of this tuple is the number of spectrograms that will be fed to the encoder.
        :param multichannel_stacked_spectrograms: If True, this dataset will multi-layer spectrograms
            (1 layer = 1 midi pitch and velocity). If False, the dataset length will be multiplied by the number
            of midi notes.
        :param n_mel_bins: Number of frequency bins for the Mel-spectrogram. If -1, the usual STFT will be used
        :param mel_fmin: TODO implement
        :param mel_fmax: TODO implement
        :param normalize_audio:  If True, audio from RenderMan will be normalized
        :param spectrogram_min_dB:  Noise-floor threshold value for log-scale spectrograms
        :param spectrogram_normalization: 'min_max' to get output spectrogram values in [-1, 1], or 'mean_std'
            to get zero-mean unit-variance output spectrograms. None to disable normalization.
        :param data_storage_path: The absolute folder to store the generated datasets.
        """
        # TODO implement audio_data_augmentation bool arg
        self.note_duration = note_duration
        self.n_fft = n_fft
        self.fft_hop = fft_hop
        self.Fs = Fs
        self.midi_notes = midi_notes
        if len(self.midi_notes) == 1:  # A 1-note dataset cannot handle multi-note stacked spectrograms
            assert not multichannel_stacked_spectrograms  # Check ctor arguments
        self._multichannel_stacked_spectrograms = multichannel_stacked_spectrograms
        self.n_mel_bins = n_mel_bins
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.normalize_audio = normalize_audio
        self._data_storage_path = pathlib.Path(data_storage_path) if data_storage_path is not None else None
        self._data_augmentation = data_augmentation
        self._random_seed = random_seed
        self._rng = np.random.default_rng(seed=self._random_seed)
        # - - - - - Attributes to be set by the child concrete class - - - - -
        self.valid_preset_UIDs = np.zeros((0,))  # UIDs (may be indexes) of valid presets for this dataset
        # - - - Spectrogram utility class - - -
        if self.n_mel_bins <= 0:
            self.compute_spectrogram = utils.audio.Spectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB)
        else:  # FIXME actually implement mel f min/max
            self.compute_spectrogram = utils.audio.MelSpectrogram(self.n_fft, self.fft_hop, spectrogram_min_dB,
                                                                  self.n_mel_bins, self.Fs)
        # spectrogram min/max/mean/std statistics: must be loaded after super() ctor (depend on child class args)
        self.spectrogram_normalization = spectrogram_normalization
        # will be automatically assigned after regeneration of spectrograms
        self.spec_stats = None  # type: Optional[dict]

    @property
    @abstractmethod
    def synth_name(self):
        pass

    def __str__(self):
        return "Dataset of {}/{} {} presets. Total items count {}: {} MIDI notes / preset, {} spectrograms.\n" \
               "{} Spectrogram items, size={}, min={:.1f}dB, normalization:{}" \
            .format(self.valid_presets_count, self.total_nb_presets, self.synth_name,
                    len(self), self.midi_notes_per_preset,
                    ('stacked' if self.midi_notes_per_preset > 1 and self._multichannel_stacked_spectrograms else 'independent'),
                    ("Linear" if self.n_mel_bins <= 0 else "Mel"), self.get_spectrogram_tensor_size(),
                    self.compute_spectrogram.min_dB, self.spectrogram_normalization)

    def __len__(self):  # Required for any torch.utils.data.Dataset
        if self._multichannel_stacked_spectrograms:
            return self.valid_presets_count
        else:
            return self.valid_presets_count * self.midi_notes_per_preset

    def __getitem__(self, i):
        """ Returns a tuple containing :
                - a 2D scaled dB spectrograms tensor (1st dim: MIDI note, 2nd dim: freq; 2rd dim: time),
                - a 1d tensor with remaining int info (preset UID, midi note, vel).
                - a 1d tensor of labels (0, 1 values)
        """
        # If several notes available but single-spectrogram output: we have to convert i into a UID and a note index
        if self.midi_notes_per_preset > 1 and not self._multichannel_stacked_spectrograms:
            preset_index = i // self.midi_notes_per_preset
            midi_note_indexes = [i % self.midi_notes_per_preset]
        else:
            preset_index = i
            midi_note_indexes = range(self.midi_notes_per_preset)
        # Load params and a list of spectrograms (1-element list is fine). 1 spectrogram per MIDI
        preset_UID = self.valid_preset_UIDs[preset_index]
        spectrograms = list()
        # TODO random noise added to spectrograms?
        # The same variation is used for all notes
        variation = self._rng.integers(0, self.nb_variations_per_note) if self._data_augmentation else 0
        for midi_note_idx in midi_note_indexes:
            midi_pitch, midi_vel = self.midi_notes[midi_note_idx]
            # Spectrogram, or Mel-Spectrogram if requested (see ctor arguments)
            spectrograms.append(torch.load(self.get_spec_file_path(preset_UID, midi_pitch, midi_vel, variation)))

        # Tuple output. Warning: torch.from_numpy does not copy values (torch.tensor(...) ctor does)
        if len(midi_note_indexes) == 1:
            ref_midi_pitch, ref_midi_velocity = self.midi_notes[midi_note_indexes[0]]
        else:
            # FIXME the MIDI pitch and velocity should be a separate tensor, for multi-layer spectrogram
            #   but this will break compatibility with much code and many notebooks
            ref_midi_pitch, ref_midi_velocity = self.midi_notes[0]
        return torch.stack(spectrograms), \
            torch.tensor([preset_UID, ref_midi_pitch, ref_midi_velocity], dtype=torch.int32), \
            self.get_labels_tensor(preset_UID)

    @property
    @abstractmethod
    def total_nb_presets(self):
        """ Total number of presets in the original database, which might be greater than the number of
        available presets in this dataset (some presets can be excluded from learning). """
        pass

    @property
    def valid_presets_count(self):
        """ Total number of presets currently available from this dataset. """
        return len(self.valid_preset_UIDs)

    def get_index_from_preset_UID(self, preset_UID):
        """ Returns the dataset index (or list of indexes) of a preset described by its UID. """
        try:
            index_in_valid_list = list(self.valid_preset_UIDs).index(preset_UID)
        except ValueError:
            raise ValueError("Preset UID {} is not a valid preset UID (it might have been excluded from this dataset)"
                             .format(preset_UID))
        # Check: are there multiple MIDI notes per preset? (dataset size artificial increase)
        if self.midi_notes_per_preset > 1 and not self._multichannel_stacked_spectrograms:  # 'annoying' case
            base_index = index_in_valid_list * self.midi_notes_per_preset
            return [base_index + i for i in range(self.midi_notes_per_preset)]
        else:  # 'usual' case: each UID has its own unique dataset index
            return index_in_valid_list

    @property
    def default_midi_note(self):
        """ Default MIDI pitch and velocity, e.g. for audio renders evaluation, labelling, ... """
        return 60, 85

    @property
    def midi_notes_per_preset(self):
        """ Number of available midi notes (different pitch and/or velocity) for a given preset. """
        return len(self.midi_notes)

    @property
    def nb_variations_per_note(self):
        """ Number of variations (data augmentation) available for a given note. """
        return 1  # Default value, to be overridden by child class if data augmentation is available

    @property
    def multichannel_stacked_spectrograms(self):
        """ If True, this dataset's spectrograms are multi-channel, each channel corresponding to a MIDI note.
         If False, this dataset's spectrograms are single-channel, but different dataset items can correspond to
         different MIDI notes. """
        return self._multichannel_stacked_spectrograms

    # ================================== Labels =================================

    def get_labels_tensor(self, preset_UID):
        """ Returns a tensor of torch.int8 zeros and ones - each value is 1 if the preset is tagged with the
        corresponding label. """
        return torch.tensor([1], dtype=torch.int8)  # 'NoLabel' is the only default label

    def get_labels_name(self, preset_UID):
        """ Returns the list of string labels assigned to a preset """
        return ['NoLabel']  # Default: all presets are tagged with this dummy label. Implement in concrete class

    @property
    def available_labels_names(self):
        """ Returns a list of string description of labels. """
        return ['NoLabel']  # this dataset allows no label

    @property
    def labels_count(self):
        return len(self.available_labels_name)

    # ================================== WAV files =================================

    @property
    def data_storage_path(self):
        """ Default storage path (e.g. for pre-rendered spectrograms) is relative to this script's folder.
        A child class should override this method to separate data from Python code. """
        if self._data_storage_path is not None:
            return self._data_storage_path
        else:
            return pathlib.Path(__file__).parent.joinpath("{}_data".format(self.synth_name))

    @abstractmethod
    def get_wav_file(self, preset_UID, midi_note, midi_velocity, variation=0):
        pass

    @abstractmethod
    def get_audio_file_stem(self, preset_UID, midi_note, midi_velocity, variation=0):
        """ Returns the name of a given audio (.wav, spectrogram, ...) file, without any extension. """
        pass

    def _get_wav_file(self, preset_UID):
        """ Returns the preset_UID audio (numpy array). MIDI note and velocity of the note are the class defaults. """
        # FIXME incompatible with future multi-MIDI notes input
        return self.get_wav_file(preset_UID, self.midi_note, self.midi_velocity)

    @property
    def nb_valid_audio_files(self):
        """ The grand total of wav files that are available or can be generated (using a synth),
        including all valid presets, all MIDI notes, and all variations of presets (data augmentation). """
        return len(self.valid_preset_UIDs) * len(self.midi_notes) * self.nb_variations_per_note

    # ================================== Spectrograms (and spectrograms' stats) =================================

    def _load_spectrogram_stats(self):
        """ To be called by the child class, after this parent class construction (because stats file path
        can depend on child class constructor arguments). """
        if self.spectrogram_normalization is not None:
            try:
                f = open(self._spectrogram_stats_file, 'r')
                self.spec_stats = json.load(f)
            except IOError:
                self.spec_stats = None
                raise FileNotFoundError("Cannot open '{}' spectrograms' stats file.".format(self._spectrogram_stats_file))
            expected_keys = ['min', 'max', 'mean', 'std']
            for k in expected_keys:
                if k not in self.spec_stats:
                    raise ValueError("Missing '{}' key in {} stats file.".format(k, self._spectrogram_stats_file))

    def get_spectrogram_tensor_size(self):
        """ Returns the size of the first tensor (2D image) returned by this dataset. """
        try:
            item = self.__getitem__(0)
            return item[0].size()
        except FileNotFoundError as e:  # __getitem__ might fail until the dataset is fully generated
            return 'Unknown'

    @property
    def _spectrograms_folder(self):
        return self.data_storage_path.joinpath("Specs_{}".format(self._spectrogram_description))

    def get_spec_file_path(self, preset_UID, midi_note, midi_velocity, variation=0):
        return self._spectrograms_folder\
            .joinpath(self.get_audio_file_stem(preset_UID, midi_note, midi_velocity, variation) + '.pt')

    @property
    def _spectrogram_stats_folder(self):
        return self.data_storage_path.joinpath("SpecStats")

    @property
    def _spectrogram_description(self):
        """ Returns a short description of the current spectrograms' characteristics,
        both readable and usable as file names. """
        specs_description = 'nfft{:04d}hop{:04d}mels'.format(self.n_fft, self.fft_hop)
        specs_description += ('None' if self.n_mel_bins <= 0 else '{:04d}'.format(self.n_mel_bins))
        return specs_description + '_norm_{}'.format('None' if self.spectrogram_normalization is None
                                                     else self.spectrogram_normalization)

    @property
    def _spectrogram_stats_file(self):
        return self._spectrogram_stats_folder.joinpath(self._spectrogram_description + '.json')

    @property
    def _spectrogram_full_stats_file(self):
        return self._spectrogram_stats_folder.joinpath(self._spectrogram_description + '_full.csv')

    def normalize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if self.spectrogram_normalization is not None and self.spec_stats is None:
            self._load_spectrogram_stats()
        if self.spectrogram_normalization == 'min_max':  # result in [-1, 1]
            return -1.0 + (spectrogram - self.spec_stats['min'])\
                   / ((self.spec_stats['max'] - self.spec_stats['min']) / 2.0)
        elif self.spectrogram_normalization == 'mean_std':
            return (spectrogram - self.spec_stats['mean']) / self.spec_stats['std']
        elif self.spectrogram_normalization is None:
            return spectrogram
        else:
            raise ValueError("Cannot perform spectrogram normalization '{}'".format(self.spectrogram_normalization))

    def denormalize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if self.spectrogram_normalization is not None and self.spec_stats is None:
            self._load_spectrogram_stats()
        if self.spectrogram_normalization == 'min_max':  # result in [-1, 1]
            return (spectrogram + 1.0) * ((self.spec_stats['max'] - self.spec_stats['min']) / 2.0)\
                   + self.spec_stats['min']
        elif self.spectrogram_normalization == 'mean_std':
            return spectrogram * self.spec_stats['std'] + self.spec_stats['mean']
        elif self.spectrogram_normalization is None:
            return spectrogram
        else:
            raise ValueError("Cannot undo spectrogram normalization '{}'".format(self.spectrogram_normalization))

    def compute_and_store_spectrograms_and_stats(self):
        """ Pre-computes and stores all spectrograms from audio files (must have been rendered previously),
         using the current spectrogram configuration.

        Also computes min,max,mean,std on all spectrograms
        Per-preset results are stored into a .csv file
        and dataset-wide averaged results are stored into a .json file

        This function must be re-run when spectrogram parameters are changed. """
        t_start = datetime.now()
        # 0) clear previous stats files
        self._init_specs_and_stats_files()
        # 1) Compute and store spectrograms and stats
        # MKL and/or PyTorch do not use hyper-threading, and it gives better results... don't use multi-proc here
        #     -> single "batch" containing the full dataset
        full_stats = self._compute_and_store_spectrograms_and_stats_batch(self.valid_preset_UIDs)
        delta_t = (datetime.now() - t_start).total_seconds()
        print("Finished generating {} spectrograms ({:.1f} min total, {:.1f}ms/spec)"
              .format(self.nb_valid_audio_files, delta_t / 60.0, 1000.0 * delta_t / self.nb_valid_audio_files))
        self._store_spectrograms_stats(full_stats)
        # 2) Normalize spectrograms
        print("Normalization of all rendered spectrogram....")
        self._normalize_spectrograms()
        # 3) Final display
        delta_t = (datetime.now() - t_start).total_seconds()
        print("{} spectrograms processed and stored as .pt files (({:.1f} min total, {:.1f}ms/spec) \n"
              "Location: {}\n"
              "Stats (before normalization) written _full.csv and .json files ({}) "
              .format(self.nb_valid_audio_files, delta_t / 60.0, 1000.0 * delta_t / self.nb_valid_audio_files,
                      self._spectrograms_folder, self._spectrogram_description))

    def _compute_and_store_spectrograms_and_stats_batch(self, preset_UIDs):
        """ Generates and stores spectrogram tensors, and returns a dict of spectrograms' stats
        using the given list of preset UIDs. """
        full_stats = {'UID': np.zeros((self.nb_valid_audio_files,), dtype=np.int),
                      'min': np.zeros((self.nb_valid_audio_files,)), 'max': np.zeros((self.nb_valid_audio_files,)),
                      'mean': np.zeros((self.nb_valid_audio_files,)), 'var': np.zeros((self.nb_valid_audio_files,))}
        i = 0
        for preset_UID in self.valid_preset_UIDs:
            for midi_note in self.midi_notes:
                pitch, vel = midi_note[0], midi_note[1]
                for variation in range(self.nb_variations_per_note):
                    x_wav, Fs = self.get_wav_file(preset_UID, pitch, vel, variation)
                    if Fs != self.Fs:
                        raise ValueError("Wrong sampling frequency ({} instead of {}} preset UID = {}"
                                         .format(Fs, self.Fs, preset_UID))
                    tensor_spectrogram = self.compute_spectrogram(x_wav)
                    torch.save(tensor_spectrogram, self.get_spec_file_path(preset_UID, pitch, vel, variation))
                    full_stats['UID'][i] = preset_UID
                    full_stats['min'][i] = torch.min(tensor_spectrogram).item()
                    full_stats['max'][i] = torch.max(tensor_spectrogram).item()
                    full_stats['var'][i] = torch.var(tensor_spectrogram).item()
                    full_stats['mean'][i] = torch.mean(tensor_spectrogram, dim=(0, 1)).item()
                    i += 1
        return full_stats

    def _init_specs_and_stats_files(self):
        if not os.path.exists(self._spectrograms_folder):
            os.makedirs(self._spectrograms_folder)
        if not os.path.exists(self._spectrogram_stats_folder):
            os.makedirs(self._spectrogram_stats_folder)
        open(self._spectrogram_stats_file, 'w').close()
        open(self._spectrogram_full_stats_file, 'w').close()

    def _store_spectrograms_stats(self, full_stats):
        """ Stats are processed after spectrograms are generated (possible multi-processed generation) """
        # Average of all columns (std: sqrt(variance avg))
        dataset_stats = {'min': full_stats['min'].min(),
                         'max': full_stats['max'].max(),
                         'mean': full_stats['mean'].mean(),
                         'std': np.sqrt(full_stats['var'].mean())}
        full_stats['std'] = np.sqrt(full_stats['var'])
        del full_stats['var']
        # Final output
        full_stats = pd.DataFrame(full_stats)
        full_stats.to_csv(self._spectrogram_full_stats_file)
        self.spec_stats = dataset_stats
        with open(self._spectrogram_stats_file, 'w') as f:
            json.dump(dataset_stats, f)

    def _normalize_spectrograms(self):
        """ Loads all pre-computed spectrograms, normalizes them using global stats,
         and writes normalized spectrogram over the pre-computed ones. """
        for preset_UID in self.valid_preset_UIDs:
            for midi_note in self.midi_notes:
                pitch, vel = midi_note[0], midi_note[1]
                for variation in range(self.nb_variations_per_note):
                    file_path = self.get_spec_file_path(preset_UID, pitch, vel, variation)
                    tensor_spectrogram = torch.load(file_path)
                    tensor_spectrogram = self.normalize_spectrogram(tensor_spectrogram)
                    torch.save(tensor_spectrogram.clone(), file_path)



class PresetDataset(AudioDataset):
    def __init__(self, note_duration,
                 n_fft, fft_hop, Fs,
                 midi_notes=((60, 100),),
                 multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=30.0, mel_fmax=11e3,
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization='min_max',
                 learn_mod_wheel_params=False
                 ):
        """
        Abstract Base Class for any synthesizer presets dataset (audio samples + associated presets).

        :param learn_mod_wheel_params: Indicates whether parameters related to the MIDI modulation wheel should
            be learned or not.
        """
        super().__init__(note_duration, n_fft, fft_hop, Fs, midi_notes, multichannel_stacked_spectrograms,
                         n_mel_bins, mel_fmin, mel_fmax, normalize_audio, spectrogram_min_dB, spectrogram_normalization)
        self.learn_mod_wheel_params = learn_mod_wheel_params
        # - - - - - Attributes to be set by the child concrete class - - - - -
        self.learnable_params_idx = list()  # Indexes of learnable VSTi params (some params may be constant or unused)

    def __str__(self):
        return "{}\n{} learnable synth params, {} fixed params." \
            .format(super().__str__(),
                    len(self.learnable_params_idx), self.total_nb_params - len(self.learnable_params_idx))

    def __getitem__(self, i):
        # TODO on-the-fly audio generation. We should try:
        #  - Use shell command to run a dedicated script. The script writes AUDIO_SAMPLE_TEMP_ID.wav
        #  - wait for the file to be generated on disk (or for the command to notify... something)
        #  - read and delete this .wav file
        spectrograms, notes_and_UID, labels = super().__getitem__(i)
        preset_UID = notes_and_UID[0].item()

        # TODO preset should be optional (for pre-training using sound and notes only)
        preset_params = self.get_full_preset_params(preset_UID)
        return spectrograms, torch.squeeze(preset_params.get_learnable(), 0), notes_and_UID, labels

    @abstractmethod
    def get_full_preset_params(self, preset_UID) -> PresetsParams:
        """ Returns a PresetsParams instance (see preset.py) of 1 preset for the requested preset_UID """
        pass

    @property
    def preset_param_names(self):
        """ Returns a List which contains the name of all parameters of presets (free and constrained). """
        return ['unnamed_param_{}'.format(i) for i in range(self.total_nb_params)]

    def get_preset_param_cardinality(self, idx, learnable_representation=True):
        """ Returns the cardinality i.e. the number of possible different values of all parameters.
        A -1 cardinal indicates a continuous parameter.
        :param idx: The full-preset (VSTi representation) index
        :param learnable_representation: Some parameters can have a reduced cardinality for learning
        (and their learnable representation is scaled consequently). """
        return -1  # Default: continuous params only

    def get_preset_param_quantized_steps(self, idx, learnable_representation=True):
        """ Returns a numpy array of possible quantized values of a discrete parameter. Quantized values correspond
        to floating-point VSTi control values. Returns None if idx refers to a continuous parameter. """
        card = self.get_preset_param_cardinality(idx, learnable_representation)
        if card == -1:
            return None
        elif card == 1:  # Constrained one-value parameter
            return np.asarray([0.5])
        elif card >= 2:
            return np.linspace(0.0, 1.0, endpoint=True, num=card)
        else:
            raise ValueError("Invalid parameter cardinality {}".format(card))

    @property
    def learnable_params_count(self):
        """ Number of learnable VSTi controls. """
        return len(self.learnable_params_idx)

    @property
    def learnable_params_tensor_length(self):
        """ Length of a learnable parameters tensor (contains single-element numerical values and one-hot encoded
        categorical params). """
        _, params, _, _ = self.__getitem__(0)  # future FIXME: corriger ça quand 
        return params.shape[0]

    @property
    def vst_param_learnable_model(self):
        """ List of models for full-preset (VSTi-compatible) parameters. Possible values are None for non-learnable
        parameters, 'num' for numerical data (continuous or discrete) and 'cat' for categorical data. """
        return ['num' for _ in range(self.total_nb_params)]  # Default: 'num' only

    @property
    def numerical_vst_params(self):
        """ List of indexes of numerical parameters (whatever their discrete number of values) in the VSTi.
        E.g. a 8-step volume param is numerical, while a LFO shape param is not (it is categorical). The
        learnable model can be different from the VSTi model. """
        return [i for i in range(self.total_nb_params)]  # Default: numerical only

    @property
    def categorical_vst_params(self):
        """ List of indexes of categorical parameters in the VSTi. The learnable model can be different
        from the VSTi model."""
        return []  # Default: no categorical params

    @property
    def params_default_values(self):
        """ Dict of default values of VSTi parameters. Not all indexes are keys of this dict (many params do not
        have a default value). """
        return {}

    @property
    @abstractmethod
    def total_nb_params(self):
        """ Total count of constrained and free VST parameters of a preset. """
        pass

    @property
    def preset_indexes_helper(self):
        """ Returns the data.preset.PresetIndexesHelper instance which helps convert full/learnable presets
        from this dataset. """
        return PresetIndexesHelper(nb_params=self.total_nb_params)  # Default: identity

    @abstractmethod
    def _render_audio(self, preset_params: Sequence, midi_note: int, midi_velocity: int):
        """ Renders audio on-the-fly and returns the computed audio waveform and sampling rate.

        :param preset_params: List of preset VST parameters, constrained (constraints from this class ctor
            args must have been applied before passing preset_params).
        """
        pass

