"""
This file allows to pre-load and use the NSynth dataset.
Data is formatted such that it is compatible with synthesizer datasets (Dexed, Surge, ...).
However, this dataset does not provide synthesis parameters because it contains acoustic sounds.
"""
import copy
import pathlib
import os
import shutil
import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import soundfile
from natsort import natsorted
from collections import OrderedDict
from typing import Optional, List

from data import abstractbasedataset


class NsynthDataset(abstractbasedataset.AudioDataset):
    def __init__(self, note_duration, n_fft, fft_hop, Fs=16000,
                 midi_notes=((60, 100),), multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=0, mel_fmax=8000,
                 normalize_audio=False, spectrogram_min_dB=-120.0,
                 spectrogram_normalization: Optional[str] = 'min_max',
                 data_storage_root_path: Optional[str] = None,
                 random_seed=0, data_augmentation=True,
                 dataset_type='full',
                 exclude_instruments_with_missing_notes=True,
                 exclude_sonic_qualities: Optional[List[str]] = None,
                 force_include_all_acoustic=True
                 ):
        """
        Class for using a downloaded NSynth dataset. Can be used by a PyTorch DataLoader.
        This dataset has a fixed size, thus a preset_UID is also an index. It is the 'instrument'
        integer value from original NSynth JSON files.

        Instruments from the 'train' folder do not overlap with 'validation' or 'test';
        however, 'validation' and 'test' contain the same instruments (with different notes in each set).

        Different instances of this class should be used to create a 'train' or a 'valid_test_merged' dataset.
        Do not use an external sampler.

        Please refer to abstractbasedataset.AudioDataset for documentation about constructor arguments.

        :param dataset_type: 'full', 'train' or 'valid_test_merged' (validation and test datasets are merged because
            they share instruments)
        :param exclude_instruments_with_missing_notes: Set to False to be able to explore the full dataset
            (but the dataset won't be usable for training)
        :param exclude_sonic_qualities: List of strings 'sonic qualities' (e.g. reverb, bright, ...) which should
            be excluded from the dataset. An instrument will be excluded if at least one note presents one of the
            given qualities.
        :param force_include_all_acoustic: If True, no acoustic instrument will be excluded from this dataset
            (even if it has a sonic quality that should exclude it).
        """
        super().__init__(note_duration, n_fft, fft_hop, Fs, midi_notes, multichannel_stacked_spectrograms, n_mel_bins,
                         mel_fmin, mel_fmax, normalize_audio, spectrogram_min_dB, spectrogram_normalization,
                         data_storage_root_path, random_seed, data_augmentation)
        if self.Fs != 16000:
            raise NotImplementedError("Resampling is not available - sampling rate must be 16 kHz")
        if dataset_type not in ['full', 'train', 'valid_test_merged']:
            raise ValueError("Invalid dataset type '{}' (must be 'full', 'train' or 'valid_test_merged')".format(dataset_type))
        if dataset_type == "valid_test_merged" and self._data_augmentation:
            warnings.warn("This 'valid_test_merged' dataset instance should not use data_augmentation (please check ctor args)")
        self.dataset_type = dataset_type
        self._excluded_qualities = exclude_sonic_qualities if exclude_sonic_qualities is not None else list()
        # try load, else dataset must be generated
        try:
            self._instru_info_df = pd.read_pickle(self._instruments_info_pickle_path)  # 1.8MB pickle file
        except FileNotFoundError:
            warnings.warn("This dataset has not been properly pre-generated. Please call regenerate_json_files() "
                          "from this dataset instance.")
            self._instru_info_df = pd.DataFrame()
        self._total_nb_presets = len(self._instru_info_df)
        if len(self._instru_info_df) > 0:
            if self.dataset_type == 'train':
                k = 'train'
            elif self.dataset_type == 'valid_test_merged':
                k = 'valid'  # valid and test are be the same... because of overlapping instruments
            else:
                k = None
            # Possibly get a sub-part of the dataset
            if k is not None:
                self._instru_info_df = self._instru_info_df[self._instru_info_df.apply(lambda row: k in row['dataset'],
                                                                                       axis=1)]
            # exclude instruments based on sonic qualities
            excluded_indexes = set()
            for q in self._excluded_qualities:
                q_idx = self.sonic_qualities_str.index(q)  # Check given qualities if this gives an error
                for idx, row in self._instru_info_df.iterrows():
                    if (not force_include_all_acoustic) \
                            or (force_include_all_acoustic and row['instrument_source_str'] != 'acoustic'):
                        if row['qualities'][q_idx] > 0:
                            excluded_indexes.add(idx)
            self._instru_info_df.drop(index=excluded_indexes, inplace=True)
            # exclude instruments with missing midi notes
            if exclude_instruments_with_missing_notes:
                excluded_indexes = list()
                for idx, row in self._instru_info_df.iterrows():
                    missing_notes = False
                    for note in self.midi_notes:
                        if note not in row['notes']:
                            missing_notes = True
                    if missing_notes:
                        excluded_indexes.append(idx)
                self._instru_info_df.drop(index=excluded_indexes, inplace=True)
            # In this dataframe: row indexes are actually preset_UIDs (fixed dataset, won't change)
            # Those indexes remain even if we delete rows (very easy to use!)
            self.valid_preset_UIDs = list(self._instru_info_df['instrument'].values)
            for UID in self.excluded_patches_UIDs:
                try:
                    self.valid_preset_UIDs.remove(UID)
                except ValueError:
                    pass  # OK if key was not found (e.g. if this is 'train', this UID could be in 'valid/test')

    @property
    def synth_name(self):
        return "NSynth"

    @property
    def total_nb_presets(self):
        return self._total_nb_presets

    def get_name_from_preset_UID(self, preset_UID: int) -> str:
        return self._instru_info_df['instrument_str'][preset_UID]  # original dataframe indexes always remain usable

    @property
    def nb_variations_per_note(self):
        return 3

    def get_wav_file(self, preset_UID, midi_note, midi_velocity, variation=0):
        # TODO data augmentation: shift audio waveform of a few ms (or small integer number of samples)
        if variation != 0:
            raise NotImplementedError()
        audio_file_name = self.get_audio_file_stem(preset_UID, midi_note, midi_velocity, variation) + '.wav'
        return soundfile.read(self._audio_symlinks_dir.joinpath(audio_file_name))

    def get_audio_file_stem(self, preset_UID, midi_note, midi_velocity, variation=0):
        # wav files are the same for all data augmentation variations
        # Preset UIDs remains dataframe indexes (even if we've deleted rows from the dataframe)
        return "{}-{:03d}-{:03d}".format(self._instru_info_df['instrument_str'][preset_UID], midi_note, midi_velocity)

    def compute_and_store_spectrograms_and_stats(self):
        if self.dataset_type != 'full':
            raise ValueError("A 'full' dataset instance must be used to generate files.")
        return super().compute_and_store_spectrograms_and_stats()

    # =================================== Labels =======================================

    @property
    def sonic_qualities_str(self):
        """ The list of string representation of all 'sonic qualities' (e.g. dark, fast_decay, ...)
        https://magenta.tensorflow.org/datasets/nsynth#note-qualities

        Those qualities might vary depending on the MIDI note (they do not remain constant for all notes of a
        given instrument, e.g. a 'bass' low note can be dark, but a high-pitched note possibly won't be).
        """
        return ['bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic',
                'nonlinear_env', 'percussive', 'reverb', 'tempo-synced']

    @property
    def instrument_families_str(self):
        """ The list of string representation of all 'instrument families' (e.g. bass, organ, ...) """
        return ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

    @property
    def instrument_sources_str(self):
        """ The list of string representation of all 'instrument sources' (e.g. acoustic, electronic, ...)
        https://magenta.tensorflow.org/datasets/nsynth#note-qualities """
        return ['acoustic', 'electronic', 'synthetic']

    # ==================== Generate new JSON files (sort all items by instrument) =================

    @property
    def _instruments_info_pickle_path(self):
        return self.data_storage_path.joinpath("instruments_info.df.pickle")

    @property
    def _audio_symlinks_base_dir(self) -> pathlib.Path:
        return self.data_storage_path.joinpath("nsynth-symlinks")

    @property
    def _audio_symlinks_dir(self) -> pathlib.Path:
        return self._audio_symlinks_base_dir.joinpath('audio')

    def regenerate_json_and_symlinks(self):
        """ Generates new JSON files to access, count and analyze dataset's elements more easily.
        Also generates symlinks to access the valid and test datasets as one. """
        if self.dataset_type != 'full':
            raise ValueError("A 'full' dataset instance must be used to generate files.")
        t_start = datetime.now()
        # Generate natural-sorted json files
        for dataset_type in ['train', 'valid', 'test']:  # FIXME reactivate when dev is finished
            self._sort_examples_json(dataset_type)  # natsort and write new files
        # Delete and make an empty symlinks folder
        if self._audio_symlinks_base_dir.exists():
            shutil.rmtree(self._audio_symlinks_base_dir)
        self._init_symlinks_folder()
        # Gather data to build the instruments_info.json file and the symlinks (to access all notes from a single dir)
        instru_info = dict()  # key: UID
        for dataset_type in ['train', 'valid', 'test']:
            dataset_dir = self.data_storage_path.joinpath("nsynth-{}".format(dataset_type))
            with open(dataset_dir.joinpath("examples_natsorted.json"), 'r') as f:
                examples = json.load(f)
                audio_dir = dataset_dir.joinpath('audio')
                for instr_name_and_note, note_dict in examples.items():
                    # 1) TODO Create symlinks for all dataset notes
                    symlink_src = audio_dir.joinpath(instr_name_and_note + '.wav')
                    symlink_dst = self._audio_symlinks_dir.joinpath(instr_name_and_note + '.wav')
                    os.symlink(symlink_src, symlink_dst)
                    # 2) Process data and add to the main dict
                    instr_name = instr_name_and_note[:-8]
                    midi_note = (note_dict['pitch'], note_dict['velocity'])
                    if note_dict['sample_rate'] != self.Fs:
                        raise ValueError("Sample rate {}Hz is different from the dataset's Hz{}"
                                         .format(note_dict['sample_rate'], self.Fs))
                    # We discard elements which do not remain constant for all notes of an instrument?
                    for k in ['pitch', 'velocity', 'note', 'note_str', 'sample_rate', 'qualities_str']:
                        del note_dict[k]
                    # Does this note belong to an instrument that we haven't seen yet?
                    if instr_name not in instru_info:
                        instru_info[instr_name] = note_dict
                        instru_info[instr_name]['notes'] = [midi_note]
                        # quality labels as np array (for easy summation). No np.int (not JSON serializable)
                        instru_info[instr_name]['qualities'] = np.asarray(note_dict['qualities'], dtype=np.int)
                        instru_info[instr_name]['dataset'] = [dataset_type]
                    else:  # if we met this instr before: check attributes
                        # we sum the nb of labels found across all notes
                        instru_info[instr_name]['qualities'] += np.asarray(note_dict['qualities'], dtype=np.int)
                        del note_dict['qualities']
                        for k in note_dict:  # Only a few keys remain at this point (the others were deleted already)
                            if instru_info[instr_name][k] != note_dict[k]:
                                raise ValueError("File '{}', note '{}': discrepancy between '{}' values"
                                                 "(expected: {}, found:{})"
                                                 .format(f.name, instr_name, k, instru_info[instr_name][k], note_dict[k]))
                        instru_info[instr_name]['notes'].append(midi_note)
                        if dataset_type not in instru_info[instr_name]['dataset']:
                            instru_info[instr_name]['dataset'].append(dataset_type)
        # 'qualities' labels: transform back into a list of ints
        for k, v in instru_info.items():
            v['qualities'] = [int(q) for q in list(v['qualities'])]  # np.int is not serializable
        # write instru info as json
        instru_info = copy.deepcopy(instru_info)  # deepcopy to prevent serialization issues
        with open(self.data_storage_path.joinpath("instruments_info.json"), 'w') as f:
            json.dump(instru_info, f)
        # convert to pandas df, save as pickle to ease future data visualizations
        instru_info_df = pd.DataFrame([v for k, v in instru_info.items()])
        instru_info_df.sort_values(by=['instrument'], inplace=True, ignore_index=True)  # this is the UID
        instru_info_df.to_pickle(self._instruments_info_pickle_path)
        # display time
        delta_t = (datetime.now() - t_start).total_seconds()
        print("NSynth JSON files and symlinks generated in {:.1f}s".format(delta_t))

    def _sort_examples_json(self, dataset_type: str):
        """ Writes sorted versions of the examples.json files (natsort of main keys). """
        examples_natsorted = OrderedDict()
        with open(self.data_storage_path.joinpath("nsynth-{}/examples.json".format(dataset_type)), 'r') as f:
            examples = json.load(f)
            new_keys = natsorted(examples.keys())
            for k in new_keys:
                examples_natsorted[k] = examples[k]
        with open(self.data_storage_path.joinpath("nsynth-{}/examples_natsorted.json".format(dataset_type)), 'w') as f:
            json.dump(examples_natsorted, f)

    def _init_symlinks_folder(self):
        self._audio_symlinks_base_dir.mkdir(parents=False, exist_ok=False)
        with open(self._audio_symlinks_base_dir.joinpath('SYMLINKS_ONLY.txt'), 'w') as f:
            f.write("The audio folder contains only symlinks to real NSynth audio files located in their "
                    "respective 'train', 'valid' or 'test' original folder.\n"
                    "This allows to access all files from a single directory.")
        self._audio_symlinks_dir.mkdir(parents=False, exist_ok=False)

