"""
Dexed VSTi audio renderer and presets database reader classes.

More information about the original DX7 paramaters:
https://www.chipple.net/dx7/english/edit.mode.html
https://djjondent.blogspot.com/2019/10/yamaha-dx7-algorithms.html
"""

import socket
import sys
import os
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool
import time
from typing import Iterable, List
import warnings
from abc import ABC, abstractmethod
from datetime import datetime

import librosa
import numpy as np
from scipy.io import wavfile
import sqlite3
import io
import pandas as pd

import pathlib

import synth.dexedpermutations
import librenderman as rm  # A symbolic link to the actual librenderman.so must be found in the current folder


# Pickled numpy arrays storage in sqlite3 DB
def adapt_array(arr):
    """ http://stackoverflow.com/a/31312102/190597 (SoulNibbler) """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("NPARRAY", convert_array)


def get_partial_presets_df(db_row_index_limits):
    """ Returns a partial dataframe of presets from the DB, limited a tuple of row indexes
    (first and last included).

    Useful for fast DB reading, because it involves a lot of unpickling which can be parallelized. """
    conn = PresetDatabaseABC._get_db_connection()
    nb_rows = db_row_index_limits[1] - db_row_index_limits[0] + 1
    presets_df = pd.read_sql_query("SELECT * FROM preset LIMIT {} OFFSET {}"
                                   .format(nb_rows, db_row_index_limits[0]), conn)
    conn.close()
    return presets_df



class PresetDatabaseABC(ABC):
    def __init__(self):
        # We also pre-load the names in order to close the sqlite DB
        conn = self._get_db_connection()
        names_df = pd.read_sql_query("SELECT * FROM param ORDER BY index_param", conn)
        conn.close()
        self._param_names = names_df['name'].to_list()

    @staticmethod
    def _get_db_path():
        return pathlib.Path(__file__).parent.joinpath('dexed_presets.sqlite')  # pkgutil would be better

    @staticmethod
    def _get_db_connection():
        db_path = PresetDatabaseABC._get_db_path()
        return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    @property
    @abstractmethod
    def nb_presets(self) -> int:
        pass

    @abstractmethod
    def get_preset_name(self, preset_UID: int) -> str:
        pass

    @property
    @abstractmethod
    def nb_params_per_preset(self) -> int:
        pass

    @property
    def param_names(self):
        return self._param_names

    @staticmethod
    def get_params_in_plugin_format(params: Iterable):
        """ Converts a 1D array of param values into an list of (idx, param_value) tuples """
        preset_values = np.asarray(params, dtype=np.double)  # np.float32 is not valid for RenderMan
        # Dexed parameters are nicely ordered from 0 to 154
        return [(i, preset_values[i]) for i in range(preset_values.shape[0])]



class PresetDatabase(PresetDatabaseABC):
    def __init__(self, num_workers=None):
        """ DEPRECATED - Opens the SQLite DB and copies all presets internally. This uses a lot of memory
        but allows easy multithreaded usage from multiple parallel dataloaders (1 db per dataloader). """
        warnings.warn("PresetDatabase class uses the original SQLite database, which is very slow and "
                      "is not up to date. Please use PresetDfDatabase instead", DeprecationWarning)
        super().__init__()
        self._db_path = self._get_db_path()
        conn = sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        # We load the full presets table (full DB is usually a few dozens of megabytes)
        self.all_presets_df = self._load_presets_df_multiprocess(conn, cur, num_workers)
        # 20 megabytes for 30 000 presets
        self.presets_mat = self.all_presets_df['pickled_params_np_array'].values
        self.presets_mat = np.stack(self.presets_mat)
        # Memory save: param values are removed from the main dataframe
        self.all_presets_df.drop(columns='pickled_params_np_array', inplace=True)
        # Algorithms are also separately stored
        self._preset_algos = self.presets_mat[:, 4]
        self._preset_algos = np.asarray(np.round(1.0 + self._preset_algos * 31.0), dtype=np.int)
        conn.close()

    def _load_presets_df_multiprocess(self, conn, cur, num_workers):
        if num_workers is None:
            num_workers = os.cpu_count() // 2
        cur.execute('SELECT COUNT(1) FROM preset')
        presets_count = cur.fetchall()[0][0]
        num_workers = np.minimum(presets_count, num_workers)
        # The last process might have a little more work to do
        rows_count_by_proc = presets_count // num_workers
        row_index_limits = list()
        for n in range(num_workers-1):
            row_index_limits.append([n * rows_count_by_proc, (n+1) * rows_count_by_proc - 1])
        # Last proc takes the remaining
        row_index_limits.append([(num_workers-1)*rows_count_by_proc, presets_count-1])
        if sys.gettrace() is not None:  # PyCharm debugger detected (should work with others)
            with ThreadPool(num_workers) as p:  # multiproc breaks PyCharm remote debug
                partial_presets_dfs = p.map(get_partial_presets_df, row_index_limits)
        else:
            with multiprocessing.Pool(num_workers) as p:
                partial_presets_dfs = p.map(get_partial_presets_df, row_index_limits)
        return pd.concat(partial_presets_dfs)

    def __str__(self):
        return "{} DX7 presets in database '{}'.".format(len(self.all_presets_df), self._db_path)

    def get_preset_values(self, idx, plugin_format=False):  # FIXME move to ABC mother class
        """ Returns a preset from the DB.

        :param idx: the preset 'row line' in the DB (not the index_preset value, which is an ID)
        :param plugin_format: if True, returns a list of (param_index, param_value) tuples. If False, returns the
            numpy array of param values. """
        preset_values = self.presets_mat[idx, :]
        if plugin_format:
            return self.get_params_in_plugin_format(preset_values)
        else:
            return preset_values

    def get_param_names(self):
        return self._param_names

    def get_preset_indexes_for_algorithms(self, algos):
        """ Returns a list of indexes of presets using the given algorithms in [[1 ; 32]] """
        indexes = []
        for i in range(self._preset_algos.shape[0]):
            if self._preset_algos[i] in algos:
                indexes.append(i)
        return indexes

    def get_size_info(self):
        """ Prints a detailed view of the size of this class and its main elements """
        main_df_size = self.all_presets_df.memory_usage(deep=True).values.sum()
        preset_values_size = self.presets_mat.size * self.presets_mat.itemsize
        return "Dexed Presets Database class size: " \
               "preset values matrix {:.1f} MB, presets dataframe {:.1f} MB"\
            .format(preset_values_size/(2**20), main_df_size/(2**20))

    @staticmethod
    def _get_presets_folder():
        return pathlib.Path(__file__).parent.absolute().joinpath('dexed_presets')

    def write_all_presets_to_files(self, verbose=True):
        """ Write all presets' parameter values to separate pickled files, for multi-processed multi-worker
        DataLoader. File names are presetXXXXXX_params.pickle where XXXXXX is the preset UID (it is not
        its row index in the SQLite database).

        Presets' names will be written to presetXXXXXX_name.txt,
        and comma-separated labels to presetXXXXXX_labels.txt.

        Performs consistency checks (e.g. labels, ...). TODO implement all consistency checks

        All files will be written to ./dexed_presets/ """
        presets_folder = self._get_presets_folder()
        if not os.path.exists(presets_folder):
            os.makedirs(presets_folder)
        for i in range(len(self.presets_mat)):
            preset_UID = self.all_presets_df.iloc[i]['index_preset']
            param_values = self.presets_mat[i, :]
            base_name = "preset{:06d}_".format(preset_UID)
            # ((un-)pickling has been done far too many times for these presets... could have been optimized)
            with open(presets_folder.joinpath(base_name + "params.pickle"), 'wb') as f:
                pickle.dump(param_values, f)
            with open(presets_folder.joinpath(base_name + "name.txt"), 'w') as f:
                f.write(self.all_presets_df.iloc[i]['name'])
            with open(presets_folder.joinpath(base_name + "labels.txt"), 'w') as f:
                labels = self.all_presets_df.iloc[i]['labels']
                labels_list = labels.split(',')
                for l in labels_list:
                    if not any([l == l_ for l_ in self.get_available_labels()]):  # Checks if any is True
                        raise ValueError("Label '{}' should not be available in self.all_presets_df".format(l))
                f.write(labels)
        if verbose:
            print("[dexed.PresetDatabase] Params, names and labels from SQLite DB written as .pickle and .txt files")

    @staticmethod
    def get_preset_params_values_from_file(preset_UID):
        warnings.warn("PresetDatabase class uses the original SQLite database, which is very slow and "
                      "is not up to date. Please use PresetDfDatabase instead", DeprecationWarning)
        return np.load(PresetDatabase._get_presets_folder()
                       .joinpath( "preset{:06d}_params.pickle".format(preset_UID)), allow_pickle=True)

    @staticmethod
    def get_preset_name_from_file(preset_UID):
        warnings.warn("PresetDatabase class uses the original SQLite database, which is very slow and "
                      "is not up to date. Please use PresetDfDatabase instead", DeprecationWarning)
        with open(PresetDatabase._get_presets_folder()
                  .joinpath( "preset{:06d}_name.txt".format(preset_UID)), 'r') as f:
            name = f.read()
        return name

    @staticmethod
    def get_available_labels():
        raise DeprecationWarning("These labels were extracted from the HPSS analysis and are obsolete.")
        return 'harmonic', 'percussive', 'sfx'

    @staticmethod
    def get_preset_labels_from_file(preset_UID):
        """ Return the preset labels as a list of strings. """
        with open(PresetDatabase._get_presets_folder()
                  .joinpath("preset{:06d}_labels.txt".format(preset_UID)), 'r') as f:
            labels = f.read()
        return labels.split(',')



class PresetDfDatabase(PresetDatabaseABC):
    def __init__(self):
        super().__init__()
        with open(PresetDfDatabase._get_dataframe_db_path(), 'rb') as f:
            self._presets_df = pickle.load(f)
        # Build UID -> local idx dict, for faster access to data by UID
        self._UID_to_local_idx = {self._presets_df.iloc[idx]['preset_UID']: idx
                                  for idx in range(len(self._presets_df))}
        # TODO get available labels

    @property
    def nb_presets(self) -> int:
        return len(self._presets_df)

    def get_preset_name(self, preset_UID: int, long_name=False) -> str:
        df_idx = self._UID_to_local_idx[preset_UID]
        name = self._presets_df.iloc[df_idx]['name']
        if long_name:
            name += ' ({})'.format(self._presets_df.iloc[df_idx]['cartridge_name'])
        return name

    def get_preset_params_values(self, preset_UID: int):
        return self._presets_df.iloc[self._UID_to_local_idx[preset_UID]]['params_values']

    @property
    def nb_params_per_preset(self) -> int:
        return self._presets_df.iloc[0]['params_values'].shape[0]

    @property
    def all_preset_UIDs(self):
        return self._presets_df['preset_UID'].values

    @staticmethod
    def _get_dataframe_db_path():
        return pathlib.Path(__file__).parent.joinpath('dexed_presets.df.pickle')

    @staticmethod
    def save_sqlite_to_df(verbose=False):
        """ Saves the reference .sqlite presets database into an equivalent pandas dataframe. """
        t_start = datetime.now()
        conn = PresetDfDatabase._get_db_connection()
        presets_df = pd.read_sql_query("SELECT * FROM preset", conn)
        cartridges_df = pd.read_sql_query("SELECT * FROM cartridge", conn)
        params_info_df = pd.read_sql_query("SELECT * FROM param", conn)  # Contains params' names
        conn.close()
        if verbose:
            print("[PresetDfDatabase] SQLite tables were read in {:.1f}s"
                  .format((datetime.now() - t_start).total_seconds()))
        # Post-processing: remove/rename SQLite columns, add cartridges names
        presets_df = presets_df.rename(columns={"index_preset": "preset_UID"})  # Corresponding Python variable name
        presets_df = presets_df.rename(columns={"pickled_params_np_array": "params_values"})
        presets_df = presets_df.drop(columns=['other_names'])
        presets_df = presets_df.rename(columns={"labels": "hpss_labels"})  # DAFx21 paper: Harmonic-Percussive labels
        cartridge_names = list()
        for cart_idx in presets_df['index_cart'].values:  # Very slow search... but done only once
            name = cartridges_df[cartridges_df['index_cart'] == cart_idx]['name'].values  # array
            if len(name) != 1:
                raise ValueError('index_cart should be unique in the database')
            cartridge_names.append(name[0])
        presets_df['cartridge_name'] = cartridge_names
        # Save the full df
        with open(PresetDfDatabase._get_dataframe_db_path(), 'wb') as f:
            pickle.dump(presets_df, f)  # Pickle Reloading takes < 0.0 ms from the SSD



class Dexed:
    """ A Dexed (DX7) synth that can be used through RenderMan for offline wav rendering. """

    def __init__(self, output_Fs, render_Fs=48000,
                 plugin_path="/home/gwendal/Jupyter/AudioPlugins/Dexed.so",
                 midi_note_duration_s=3.0, render_duration_s=4.0,
                 buffer_size=512, fft_size=512,
                 fadeout_duration_s=0.0,  # Default: disabled
                 ):
        self.fadeout_duration_s = fadeout_duration_s  # To reduce STFT discontinuities with long-release presets
        self.midi_note_duration_s = midi_note_duration_s
        self.render_duration_s = render_duration_s

        self.plugin_path = plugin_path
        self.render_Fs = render_Fs
        self.reduced_Fs = output_Fs
        self.buffer_size = buffer_size
        self.fft_size = fft_size  # FFT not used

        self.engine = rm.RenderEngine(self.render_Fs, self.buffer_size, self.fft_size)
        self.engine.load_plugin(self.plugin_path)

        # A generator preset is a list of (int, float) tuples.
        self.preset_gen = rm.PatchGenerator(self.engine)  # 'RenderMan' generator
        self.current_preset = None

    def __str__(self):
        return "Plugin loaded from {}, Fs={}Hz (output downsampled to {}Hz), buffer {} samples."\
               "MIDI note on duration: {:.1f}s / {:.1f}s total."\
            .format(self.plugin_path, self.render_Fs, self.reduced_Fs, self.buffer_size,
                    self.midi_note_duration_s, self.render_duration_s)

    def render_note(self, midi_note, midi_velocity, normalize=False):
        """ Renders a midi note (for the currently set patch) and returns the (normalized) float array and
         associated sampling rate. """
        self.engine.render_patch(midi_note, midi_velocity, self.midi_note_duration_s, self.render_duration_s)
        audio_out = self.engine.get_audio_frames()
        audio = np.asarray(audio_out)
        fadeout_len = int(np.floor(self.render_Fs * self.fadeout_duration_s))
        if fadeout_len > 1:  # fadeout might be disabled if too short
            fadeout = np.linspace(1.0, 0.0, fadeout_len)
            audio[-fadeout_len:] = audio[-fadeout_len:] * fadeout
        if normalize:
            audio = audio * (0.99 / np.abs(audio).max())  # to prevent 16-bit conversion clipping
        audio = librosa.resample(audio, self.render_Fs, self.reduced_Fs, res_type="kaiser_best")
        return audio, self.reduced_Fs

    def assign_preset(self, preset):
        """ :param preset: List of tuples (param_idx, param_value) """
        self.current_preset = preset
        self.engine.set_patch(self.current_preset)

    def assign_random_preset_short_release(self):
        """ Generates a random preset with a short release time - to ensure a limited-duration
         audio recording, and configures the rendering engine to use that preset. """
        self.current_preset = dexed.preset_gen.get_random_patch()
        self.set_release_short()
        self.engine.set_patch(self.current_preset)

    def set_release_short(self, eg_4_rate_min=0.5):
        raise AssertionError()  # deprecated - should return the modified params as well
        for i, param in enumerate(self.current_preset):
            idx, value = param  # a param is actually a tuple...
            # Envelope release level: always to zero (or would be an actual hanging note)
            if idx == 30 or idx == 52 or idx == 74 or idx == 96 or idx == 118 or idx == 140:
                self.current_preset[i] = (idx, 0.0)
            # Envelope release time: quite short (higher float value: shorter release)
            elif idx == 26 or idx == 48 or idx == 70 or idx == 92 or idx == 114 or idx == 136:
                self.current_preset[i] = (idx, max(eg_4_rate_min, value))
        self.engine.set_patch(self.current_preset)

    def set_default_general_filter_and_tune_params(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        assert self.current_preset is not None
        self.current_preset[0] = (0, 1.0)  # filter cutoff
        self.current_preset[1] = (1, 0.0)  # filter reso
        self.current_preset[2] = (2, 1.0)  # output vol
        self.current_preset[3] = (3, 0.5)  # master tune
        self.current_preset[13] = (13, 0.5)  # Sets the 'middle-C' note to the default C3 value
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def set_default_general_filter_and_tune_params_(preset_params):
        """ Modifies some params in-place for the given numpy array """
        preset_params[[0, 1, 2, 3, 13]] = np.asarray([1.0, 0.0, 1.0, 0.5, 0.5])

    def set_all_oscillators_on(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        assert self.current_preset is not None
        for idx in [44, 66, 88, 110, 132, 154]:
            self.current_preset[idx] = (idx, 1.0)
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def set_all_oscillators_on_(preset_params):
        """ Modifies some params of the given numpy array to ensure that all operators (oscillators) are ON.
        Data is modified in place. """
        preset_params[[44, 66, 88, 110, 132, 154]] = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    @staticmethod
    def set_all_oscillators_off_(preset_params):
        """ Modifies some params of the given numpy array to ensure that all operators (oscillators) are OFF.
        Data is modified in place. """
        preset_params[[44, 66, 88, 110, 132, 154]] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def set_oscillators_on_(preset_params, operators_to_turn_on):
        """ Modifies some params of the given numpy array to turn some operators ON. Data is modified in place.

        :param preset_params: Numpy Array of preset parameters values.
        :param operators_to_turn_on: List of integers in [1, 6]
        """
        Dexed.set_all_oscillators_off_(preset_params)
        for op_number in operators_to_turn_on:
            preset_params[44 + 22 * (op_number-1)] = 1.0

    def prevent_SH_LFO(self):
        """ If the LFO Wave is random S&H, transforms it into a square LFO wave to get deterministic
        results. Internally sets the modified preset, and returns the list of parameter values.  """
        if self.current_preset[12][1] > 0.95:  # S&H wave corresponds to a 1.0 param value
            self.current_preset[12] = (12, 4.0 / 5.0)  # Square wave is number 4/6
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def prevent_SH_LFO_(preset_params):
        """ Modifies some params in-place for the given numpy array """
        if preset_params[12] > 0.95:
            preset_params[12] = 4.0 / 5.0

    @staticmethod
    def get_midi_key_related_param_indexes():
        """ Returns a list of indexes of all DX7 parameters that apply a modulation depending on the MIDI key
        (note and/or velocity). These will be very hard to learn without providing multiple-notes input
        to the encoding network. """
        # (6. 'OSC KEY SYNC' (LFO) does NOT depend on the midi note (it syncs or not LFO phase on midi event).)
        # All the KEY L/R stuff (with breakpoint at some MIDI note) effects are really dependant on the MIDI key.
        # 36. breakpoint. Values 0 to 99 correspond to MIDI notes 9 to 108 (A-1 to C8)
        # 37/38: L/R scale (curve) depth (-> EG level scaling only?)
        # 39/40: L/R scale (=curve) type: +/-lin or +/-exp. (-> EG level scaling only?)
        # 41: rate scaling (-> EG rate scaling, longer decay for bass notes)
        # 43: key velocity (-> general OP amplitude increases(?) with MIDI velocity)
        return sorted([(36 + 22*i) for i in range(6)]\
            + [(37 + 22*i) for i in range(6)] + [(38 + 22*i) for i in range(6)]\
            + [(39 + 22*i) for i in range(6)] + [(40 + 22*i) for i in range(6)] \
            + [(41 + 22 * i) for i in range(6)] + [(43 + 22 * i) for i in range(6)])

    @staticmethod
    def get_mod_wheel_related_param_indexes():
        """ Returns a list of indexes of all DX7 parameters that influence sound depending on the MIDI
        mod wheel. These should always be learned because they are also related to LFO modulation
        (see https://fr.yamaha.com/files/download/other_assets/9/333979/DX7E1.pdf page 26) """
        # OPx A MOD SENS + Pitch general mod sens
        return [(42 + 22*i) for i in range(6)] + [14]

    @staticmethod
    def get_param_cardinality(param_index):
        """ Returns the number of possible values for a given parameter. """
        if param_index == 4:  # Algorithm
            return 32
        elif param_index == 5:  # Feedback
            return 8
        elif param_index == 6:  # OSC key sync (off/on)
            return 2
        elif param_index == 11:  # LFO key sync (off/on)
            return 2
        elif param_index == 12:  # LFO wave
            return 6
        elif param_index == 14:  # pitch modulation sensitivity
            return 8
        elif param_index >= 23:  # oscillators (operators) params
            if (param_index % 22) == (32 % 22):  # OPx Mode (ratio/fixed)
                return 2
            elif (param_index % 22) == (33 % 22):  # OPx F coarse
                return 32
            elif (param_index % 22) == (35 % 22):  # OPx OSC Detune
                return 15
            elif (param_index % 22) == (39 % 22):  # OPx L Key Scale (-lin, -exp, +exp, +lin)
                return 4
            elif (param_index % 22) == (40 % 22):  # OPx R Key Scale (-lin, -exp, +exp, +lin)
                return 4
            elif (param_index % 22) == (41 % 22):  # OPx Rate Scaling
                return 8
            elif (param_index % 22) == (42 % 22):  # OPx A modulation sensitivity
                return 4
            elif (param_index % 22) == (43 % 22):  # OPx Key Velocity
                return 8
            elif (param_index % 22) == (44 % 22):  # OPx Switch (off/on)
                return 2
            else:  # all other are 'continuous' but truly present 100 steps
                return 100
        else:
            return 100

    @staticmethod
    def get_numerical_params_indexes():
        indexes = [0, 1, 2, 3, 5,  # cutoff, reso, output, master tune, feedback (card:8)
                   7, 8, 9, 10,  # lfo speed, lfo delay (before LFO actually modulates), lfo pm depth, lfo am depth
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # transpose, pitch mod sensitivity, pitch EG rates/levels
        for i in range(6):  # operators
            for j in [23, 24, 25, 26, 27, 28, 29, 30]:  # rates and levels
                indexes.append(j + 22*i)
            indexes.append(31 + 22*i)  # output level
            indexes.append(33 + 22*i)  # freq coarse
            indexes.append(34 + 22*i)  # freq fine
            indexes.append(35 + 22*i)  # detune (these 3 parameters kind of overlap...)
            indexes.append(36 + 22*i)  # L/R scales breakpoint
            indexes.append(37 + 22*i)  # L scale depth
            indexes.append(38 + 22*i)  # R scale depth
            indexes.append(41 + 22*i)  # rate scaling (card:8)
            indexes.append(42 + 22*i)  # amplitude mod sensitivity (card:4)
            indexes.append(43 + 22*i)  # key velocity (card:8)
        return indexes

    @staticmethod
    def get_categorical_params_indexes():
        indexes = [4, 6, 11, 12]  # algorithm, osc key sync, lfo key sync, lfo wave
        for i in range(6):  # operators
            indexes.append(32 + 22*i)  # mode (ratio or fixed frequency)
            indexes.append(39 + 22*i)  # L scale
            indexes.append(40 + 22*i)  # R scale
            indexes.append(44 + 22*i)  # op on/off switch
        return indexes

    @staticmethod
    def get_op_output_level_indices():
        return [31 + 22*i for i in range(6)]

    @staticmethod
    def get_L_R_scale_indices():
        return [39 + 22*i for i in range(6)] + [40 + 22*i for i in range(6)]

    @staticmethod
    def get_operators_params_indexes_groups() -> List[range]:
        """ Returns a list of 6 ranges, where each range contains the indices of all parameters of an operator. """
        return [range(23 + 22 * i, 23 + 22 * (i+1)) for i in range(6)]

    @staticmethod
    def get_algorithms_and_oscillators_permutations(algo: int, feedback: bool):
        return synth.dexedpermutations.get_algorithms_and_oscillators_permutations(algo, feedback)

    @staticmethod
    def get_similar_preset(preset: np.ndarray, variation: int, learnable_indices: List[int], random_seed=0):
        """ Data augmentation method: returns a slightly modified preset, which is (hopefully) quite similar
            to the input preset. """
        if variation == 0:
            return preset
        rng = np.random.default_rng((random_seed + 987654321 * variation))
        # First: change algorithm to a similar one
        preset = synth.dexedpermutations.change_algorithm_to_similar(preset, variation, random_seed)
        # Then: change a few learned parameters if variation > 1: algorithm is the hardest parameter to learn,
        # so we provide a special data augmentation for this parameter only.
        if variation == 1:
            return preset
        # If variation >= 2: random noise applied to most of parameters
        cat_indexes = Dexed.get_categorical_params_indexes()
        # don't choose a limited subset of param to augment, but use a <1.0 noise std
        for idx in learnable_indices:
            if idx == 4:  # algorithm
                pass
            # cat params: depends
            elif idx in cat_indexes:
                card = Dexed.get_param_cardinality(idx)
                # General cat params: key sync, lfo sync, lfo wave: 100% randomization
                if idx in [6, 11, 12]:
                    preset[idx] = rng.integers(0, card) / (card - 1.0)
                # OP mode: quite risky to change it... (likely to lead to inaudible/unlikely sounds)
                elif idx > 32 and ((idx - 32) % 22 == 0):
                    pass
                # L/R scales: invert lin/exp (keep +/- sign)
                elif idx in Dexed.get_L_R_scale_indices():
                    if preset[idx] < 0.5:
                        preset[idx] = rng.integers(0, 1, endpoint=True) / 3
                    else:
                        preset[idx] = rng.integers(2, 3, endpoint=True) / 3
            # "continuous" (discrete ordinal) params: triangle or gaussian noise, std depends on cardinality
            else:
                card = Dexed.get_param_cardinality(idx)
                if card < 100:  # Discrete ordinal params with a few values: smaller probability of change
                    noise = rng.choice([-1, 0, 1], p=[0.05, 0.9, 0.05]) / (card - 1.0)
                    lol = 0
                else:
                    # Noise with 1 discrete increment of 0.5 standard deviation
                    noise = rng.normal(0.0, 0.5 / (card - 1.0))
                    # Very small volume values: only positive noise
                    if idx in Dexed.get_op_output_level_indices() and preset[idx] < 0.1:
                        noise = np.abs(noise)
                preset[idx] += noise
        return np.clip(preset, 0.0, 1.0)


if __name__ == "__main__":

    test_SQLite_database = False
    test_df_database = True


    if test_df_database:
        if False:  # Convert SQLite to Dataframe
            PresetDfDatabase.save_sqlite_to_df(verbose=True)

        df_db = PresetDfDatabase()
        print(df_db)


    if test_SQLite_database:
        print("Machine: '{}' ({} CPUs)".format(socket.gethostname(), os.cpu_count()))

        t0 = time.time()
        dexed_db = PresetDatabase()
        print("{} (loaded in {:.1f}s)".format(dexed_db, time.time() - t0))
        names = dexed_db.get_param_names()
        #print("Labels example: {}".format(dexed_db.get_preset_labels_from_file(3)))

        print("numerical VSTi params: {}".format(Dexed.get_numerical_params_indexes()))
        print("categorical VSTi params: {}".format(Dexed.get_categorical_params_indexes()))

        # Compute the total number of logits, if all param are learned as categorical (full-resolution)
        logits_count = 0
        for _ in Dexed.get_numerical_params_indexes():
            logits_count += 100
        for i in Dexed.get_categorical_params_indexes():
            logits_count += Dexed.get_param_cardinality(i)
        print("{} logits, if all param are learned as categorical (full-resolution)".format(logits_count))

        # ***** RE-WRITE ALL PRESETS TO SEPARATE PICKLE/TXT FILES *****
        if False:
            # Approx. 360Mo (yep, the SQLite DB is much lighter...) for all params values + names + labels
            dexed_db.write_all_presets_to_files()

        if False:
            # Test de lecture des fichiers pickled - pas besoin de la DB lue en entier
            preset_values = PresetDatabase.get_preset_params_values_from_file(0)
            preset_name = PresetDatabase.get_preset_name_from_file(0)
            print(preset_name)

        if False:
            # Test du synth lui-même
            dexed = Dexed()
            print(dexed)
            print("Plugin params: ")
            print(dexed.engine.get_plugin_parameters_description())

            #dexed.assign_random_preset_short_release()
            #pres = dexed.preset_db.get_preset_values(0, plugin_format=True)
            #dexed.assign_preset_from_db(100)
            #print(dexed.current_preset)

            #dexed.render_note(57, 100, filename="Test.wav")

            print("{} presets use algo 5".format(len(dexed_db.get_preset_indexes_for_algorithm(5))))


