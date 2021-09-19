"""
This file focuses on audio rendering and patches retrieval and storage.
Audio files are stored by SurgeDataset.

This file can be run as main script to update the JSON list of Surge patches available on the current system.
"""


# Quite dirty.... from https://github.com/surge-synthesizer/surge-python/blob/main/api-tutorial
import sys
sys.path.append( "/home/gwendal/Jupyter/AudioPlugins/surge_build" )
import surgepy  # Must be properly built and available from the folder above

import numpy as np
import pathlib
import json
from enum import IntEnum
import copy
import warnings

import librosa
from natsort import natsorted



class FxBypassLevel(IntEnum):  # 0: off, 1: send, 2: send+master, 3: all
    OFF = 0,
    SEND = 1,
    SEND_AND_MASTER = 2,
    ALL = 3



class Surge:
    def __init__(self, render_Fs=48000, reduced_Fs=16000,
                 midi_note_start_s=0.0, midi_note_duration_s=3.0, render_duration_s=4.0,
                 fx_bypass_level=FxBypassLevel.ALL):
        """
        A class for rendering notes using the Surge synthesizer through surgepy.
        Each note will be rendered using a fresh surgepy Surge instance, but this seems to be much faster
        than instantiating a new Dexed VST instance through Renderman.

        TODO add argument for note start randomization (data augmentation), here or in render_note

        :param render_Fs: Sampling rate for note rendering
        :param reduced_Fs: If >= 0, output waveforms will be downsampled after rendering. Useful for compatibility
            with other datasets (e.g. NSynth 16kHz)
        """
        self.version = surgepy.getVersion()  # for easy serialization
        self.midi_note_start_s = midi_note_start_s
        self.midi_note_duration_s = midi_note_duration_s
        self.render_duration_s = render_duration_s
        self.reduced_Fs = reduced_Fs
        self.render_Fs = render_Fs
        self.fx_bypass_level = fx_bypass_level
        # JSON list of patches (with UID, author, instrument category, patch name)
        if pathlib.Path.exists(self.get_patches_json_path()):
            with open(self.get_patches_json_path(), 'r') as f:
                self._patches_list = json.load(f)
        else:
            raise FileNotFoundError("The JSON list of patches must be created before creating a Surge instance"
                                    " (run this file as main script)")
        # Check UIDs ordering (if JSON loading changes the ordering of list's elements...?)
        last_UID = -1
        for _, patch in enumerate(self._patches_list):
            assert(last_UID < patch['UID'])  # UIDs must be strictly increasing
            last_UID = patch['UID']

    def __str__(self):
        return "Surge/surgepy {}. Note start, note duration, render duration = {:.3f}s, {:.1f}s, {:.1f}s @ {}Hz {}"\
            .format(self.version, self.midi_note_start_s, self.midi_note_duration_s, self.render_duration_s,
                    self.render_Fs, ("" if self.reduced_Fs <= 0 else "(downsampled to {}Hz)".format(self.reduced_Fs)))

    @property
    def dict_description(self):
        """ Returns a serialized description of this instance's main parameters (patches excluded) """
        d = copy.deepcopy(self.__dict__)
        del d['_patches_list']
        return d

    @staticmethod
    def get_patches_json_path():
        return pathlib.Path(__file__).parent.joinpath('surge_patches_list.json')

    def get_patch_info(self, patch_index):
        """ Returns a copy of the dict containing information about a patch. """
        return copy.deepcopy(self._patches_list[patch_index])

    def get_UID_from_index(self, patch_index):
        return self._patches_list[patch_index]['UID']

    def check_json_patches_list(self):
        pass  # TODO check patches (all UIDs must correspond to a valid folder and subfolder)

    @staticmethod
    def update_patches_list(render_Fs=48000):
        """
        Reads all 'factory' and '3rd party' presets on the current computer.
        Also assigns a UID to each one, and stores the list of presets into a surge_patches.json file.

        Expected directories structures are:
        _ patches_factory (considered as an author's name)
          |_ instrument_category (e.g. Brass)
             |_ patch_name.fxp
        _ patches_3rdparty
          |_ author_name (e.g. Argitoth)
             |_ instrument_category (e.g. Winds)
                |_ patch_name.fxp
        """
        s = surgepy.createSurge(render_Fs)
        base_path = pathlib.Path(s.getFactoryDataPath())  # actually contains factory and 3rd party patches
        patches_subdirs = ['patches_factory', 'patches_3rdparty']
        patches_list = list()
        for subdir_str in patches_subdirs:
            patch_source_folder = base_path.joinpath(subdir_str)
            if subdir_str == 'patches_factory':
                author_dirs = [patch_source_folder]
            else:
                author_dirs = natsorted([x for x in patch_source_folder.iterdir() if x.is_dir()])
            for author_dir in author_dirs:
                instrument_dirs = natsorted([x for x in author_dir.iterdir() if x.is_dir()])
                for instr_dir in instrument_dirs:
                    patch_files = natsorted(instr_dir.glob('*.fxp'))
                    for patch_file in patch_files:
                        patches_list.append({'UID': len(patches_list), 'author': author_dir.stem,
                                             'instrument_category': instr_dir.stem, 'patch_name': patch_file.stem})
        print("Found {} patches in {}".format(len(patches_list), patches_subdirs))
        with open(Surge.get_patches_json_path(), 'w') as f:
            json.dump(patches_list, f)
            print("Patches written to {}".format(Surge.get_patches_json_path()))

    @property
    def n_patches(self):
        return len(self._patches_list)

    def get_patch_path(self, s, patch_index):
        p = self._patches_list[patch_index]
        base_path = pathlib.Path(s.getFactoryDataPath())  # actually contains factory and 3rd party patches
        if p['author'] != 'patches_factory':
            base_path = base_path.joinpath('patches_3rdparty')
        patch_path = base_path.joinpath(p['author']).joinpath(p['instrument_category'])
        return patch_path.joinpath("{}.fxp".format(p['patch_name']))

    def get_synth_and_patch(self, patch_index):
        """ Creates a surge synth instance, loads a patch, and returns both """
        s = surgepy.createSurge(self.render_Fs)
        patch_path = self.get_patch_path(s, patch_index)
        # load preset and
        s.loadPatch(str(patch_path))
        return s, s.getPatch()

    @staticmethod
    def print_param_info(synth_instance, param):
        print("Current value: {}".format(synth_instance.getParamVal(param)))
        print("Min: {}".format(synth_instance.getParamMin(param)))
        print("Max: {}".format(synth_instance.getParamMax(param)))
        print("Default: {}".format(synth_instance.getParamDef(param)))
        print("ValType: {}".format(synth_instance.getParamValType(param)))
        print("Display: {}".format(synth_instance.getParamDisplay(param)))

    @property
    def nb_variations_per_note(self):
        """ Returns the number of available variations of a single note audio file (data augmentation). """
        return 3 * 2 * 3  # 3x start delays, 2x durations, 3x scene A pitch detune

    def _decode_variation_index(self, variation):
        """
        Transforms a variation index ( in [0, nb_variations_per_note[ ) into a tuple of int values

        :returns: start_delay_variation (3 vars), duration_variation (2 vars), pitch_variation (3 vars)
        """
        if variation >= self.nb_variations_per_note:
            raise ValueError("The variation index for data augmentation must be < {}"
                             .format(self.nb_variations_per_note))
        var0 = variation // 6
        r = variation - 6 * var0
        var1 = r // 3
        r -= 3 * var1
        return var0, var1, r

    def render_note(self, patch_index, midi_pitch, midi_vel, midi_ch=0, variation=0):
        """
        Creates a new Surge synth instance and renders a note using the given patch.
        :param variation: The index of a given variation for data augmentation. 0 is no variation.
        :returns: (downsampled L-channel, sampling frequency)
        """
        s, s_patch = self.get_synth_and_patch(patch_index)
        # set FX Bypass level
        s.setParamVal(s_patch['fx_bypass'], int(self.fx_bypass_level))

        start_delay_variation, duration_variation, pitch_variation = self._decode_variation_index(variation)
        # random variations (data augmentation) using (patch_index+variation) as random seed init
        rng = np.random.default_rng(seed=patch_index + variation)
        random_duration_blocks = rng.choice([-1, 1]) * duration_variation
        if pitch_variation == 0:
            random_pitch = 0.0
        else:
            random_pitch = (0.05 + 0.1 * rng.random()) * (-1.0 if pitch_variation == 2 else 1.0)

        # data augmentation: scene A pitch - small detune
        original_pitch = s.getParamVal(s_patch['scene'][0]['pitch'])
        s.setParamVal(s_patch['scene'][0]['pitch'], original_pitch + random_pitch)
        # print("(scene0pitch): {} ---randomized---> {}".format(original_pitch,
        #                                                       s.getParamVal(s_patch['scene'][0]['pitch'])))

        # Main buffer
        block_size = s.getBlockSize()
        n_blocks = int(self.render_duration_s * self.render_Fs / block_size)
        buf = s.createMultiBlock(n_blocks)
        t_block_s = block_size / self.render_Fs
        # Note on/off blocks
        note_on_block = int(round(self.midi_note_start_s / t_block_s))
        note_on_block += start_delay_variation  # Note on delay: +0, +1 or +2 blocks
        note_off_block = note_on_block + int(round(self.midi_note_duration_s / t_block_s)) + random_duration_blocks
        if note_off_block >= n_blocks:
            warnings.warn("Note off time has been limited to the last render block. Please ensure that notes do not"
                          " end before the last audio rendering block.")
            note_off_block = n_blocks - 1

        # 3-steps audio render
        # surgepy Doc for version 1.9.0.91069f8d
        # processMultiBlock(self: surgepy.SurgeSynthesizer, val: numpy.ndarray[numpy.float32], startBlock: int = 0, nBlocks: int = -1) -> None
        # Run the surge engine for multiple blocks, updating the value in the numpy array.
        # Either populate the entire array, or starting at startBlock position in the output, populate nBlocks.
        if note_on_block > 0:
            s.processMultiBlock(buf, 0, note_on_block)
        # playNote(self: surgepy.SurgeSynthesizer, channel: int, midiNote: int, velocity: int, detune: int = 0) -> None
        # Trigger a note on this Surge instance.
        s.playNote(midi_ch, midi_pitch, midi_vel)
        s.processMultiBlock(buf, note_on_block, note_off_block)
        # releaseNote(self: surgepy.SurgeSynthesizer, channel: int, midiNote: int, releaseVelocity: int = 0) -> None
        # Release a note on this Surge instance.
        s.releaseNote(midi_ch, midi_pitch)
        s.processMultiBlock(buf, note_off_block)

        # downsampling, mono (careful, possible phasing effects -> L only) and return
        buf = librosa.resample(buf[0], self.render_Fs, self.reduced_Fs, res_type="kaiser_best")
        return buf, self.reduced_Fs



if __name__ == "__main__":

    #Surge.update_patches_list()

    surge_synth = Surge()
    print(surge_synth)

    for i in range(surge_synth.nb_variations_per_note):
        print(surge_synth._decode_variation_index(i))

    surge_synth.render_note(432, 60, 100)

    d = surge_synth.dict_description
    pass
