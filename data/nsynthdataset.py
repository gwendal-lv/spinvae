"""
This file allows to pre-load and use the NSynth dataset.
Data is formatted such that it is compatible with synthesizer datasets (Dexed, Surge, ...).
However, this dataset does not provide synthesis parameters because it contains acoustic sounds.
"""
import copy
import json
from typing import Optional

import numpy as np
from natsort import natsorted
from collections import OrderedDict

import torchaudio

from data import abstractbasedataset


class NsynthDataset(abstractbasedataset.AudioDataset):
    def __init__(self, note_duration, n_fft, fft_hop, Fs=16000,
                 midi_notes=((60, 100),), multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=0, mel_fmax=8000,
                 normalize_audio=False, spectrogram_min_dB=-120.0,
                 spectrogram_normalization: Optional[str] = 'min_max',
                 data_storage_root_path: Optional[str] = None,
                 random_seed=0, data_augmentation=True
                 ):
        """
        Class for using a downloaded NSynth dataset. Can be used by a PyTorch DataLoader.

        Instruments from the 'train' folder do not overlap with 'validation' or 'test';
        however, 'validation' and 'test' contain the same instruments (with different notes in each set).

        Please refer to abstractbasedataset.AudioDataset for documentation about constructor arguments.
        """
        super().__init__(note_duration, n_fft, fft_hop, Fs, midi_notes, multichannel_stacked_spectrograms, n_mel_bins,
                         mel_fmin, mel_fmax, normalize_audio, spectrogram_min_dB, spectrogram_normalization,
                         data_storage_root_path, random_seed, data_augmentation)
        if self.Fs != 16000:
            raise NotImplementedError("Resampling is not available - sampling rate must be 16 kHz")

    @property
    def synth_name(self):
        return "NSynth"

    @property
    def total_nb_presets(self):
        pass

    def get_name_from_preset_UID(self, preset_UID: int) -> str:
        pass

    def get_wav_file(self, preset_UID, midi_note, midi_velocity, variation=0):
        pass

    def get_audio_file_stem(self, preset_UID, midi_note, midi_velocity, variation=0):
        pass   # TODO wav files are the same for all data augmentation variations

    # =================================== Labels =======================================
    @property
    def sonic_qualities_str(self):
        """ The list of string representation of all 'sonic qualities' (e.g. dark, fast_decay, ...)
        https://magenta.tensorflow.org/datasets/nsynth#note-qualities

        Those qualities might vary depending on the MIDI note (they do not remain constant for all notes of a
        given instrument, e.g. a 'bass' low note can be dark, but a high-pitched note possibly won't be).
        """
        return ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

    @property
    def instrument_families_str(self):
        """ The list of string representation of all 'instrument families' (e.g. bass, organ, ...)
        https://magenta.tensorflow.org/datasets/nsynth#note-qualities """
        return ['bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic',
                'nonlinear_env', 'percussive', 'reverb', 'tempo-synced']

    @property
    def instrument_sources_str(self):
        """ The list of string representation of all 'instrument sources' (e.g. acoustic, electronic, ...)
        https://magenta.tensorflow.org/datasets/nsynth#note-qualities """
        return ['acoustic', 'electronic', 'synthetic']

    # ==================== Generate new JSON files (sort all items by instrument) =================
    def regenerate_json_files(self):
        """ Generates new JSON files to access, count and analyze dataset's elements more easily. """
        # for dataset_type in ['train', 'valid', 'test']:  # FIXME reactivate when dev is finished
        #     self._sort_examples_json(dataset_type)
        # Gather data to build the instruments_info.json file
        instru_info = dict()  # key: UID
        for dataset_type in ['train', 'valid', 'test']:
            with open(self.data_storage_path.joinpath("nsynth-{}/examples_natsorted.json".format(dataset_type)), 'r')\
                    as f:
                examples = json.load(f)
                for instr_name_and_note, note_dict in examples.items():
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
        with open(self.data_storage_path.joinpath("instruments_info.json".format(dataset_type)), 'w') as f:
            json.dump(copy.deepcopy(instru_info), f)  # deepcopy to prevent serialization issues

        # TODO convert to pandas df, save as pickle to ease future data visualizations

        # TODO check 'instrument' field -> might become the UID?
        #   then UID should become the new key

        # TODO display time


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


# TODO sub-dataset train/valid/test must be constructed from this file

