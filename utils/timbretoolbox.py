import copy
import multiprocessing.pool
import pickle
import queue
import subprocess
import pathlib
import threading
import time
import warnings
from datetime import datetime
from typing import Optional, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ToolboxLogger(ABC):
    @abstractmethod
    def log_and_print(self, log_str: str, erase_file=False, force_print=False):
        pass


class TimbreToolboxProcess:
    def __init__(self, timbre_toolbox_path: pathlib.Path, directories_list_file: pathlib.Path, verbose=True,
                 logger: Optional[ToolboxLogger] = None, process_index: Optional[int] = None):
        """
        Runs the TimbreToolbox to process folders of audio files (folders' paths given in a separate text file).
        The 'matlab' command must be available system-wide.

        :param timbre_toolbox_path: Path the TimbreToolbox https://github.com/VincentPerreault0/timbretoolbox
        :param directories_list_file: Text file which contains a list of directories to be analyzed by the toolbox.
        :param verbose:
        """
        self.process_index = process_index
        self.logger = logger
        self.verbose = verbose
        self.current_path = pathlib.Path(__file__).parent
        self.matlab_commands = "addpath(genpath('{}')); " \
                               "cd '{}'; " \
                               "timbre('{}'); " \
                               "exit " \
            .format(str(timbre_toolbox_path),
                    str(self.current_path),  # Path to the local timbre.m file
                    str(directories_list_file)
                    )
        self.continue_queue_threads = False

    def _get_process_str(self):
        return '' if self.process_index is None else ' #{}'.format(self.process_index)

    def _enqueue_std_output(self, std_output, q: queue.Queue):
        """
        To be launched as a Thread (contains a blocking readline() call)
        Related question: https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python

        :param std_output: std::cout or std::cerr
        """
        while self.continue_queue_threads:
            line = std_output.readline()
            if line:
                q.put(line)

    def run(self):

        # Matlab args From https://arc.umich.edu/software/matlab/
        # and https://stackoverflow.com/questions/38723138/matlab-execute-script-from-command-linux-line/38723505
        proc_args = ['matlab', '-nodisplay', '-nodesktop', '-nosplash', '-r', self.matlab_commands]
        log_str = '============ Launching matlab commands (will block if a Matlab error happens) ============\n' \
                  '{}\n' \
                  'Subprocess args: {}\n'.format(datetime.now().strftime("%Y/%m/%d, %H:%M:%S"), proc_args)
        if self.process_index is not None:
            log_str = '[#{}]'.format(self.process_index) + log_str
        self._log_and_print(log_str)

        # Poll process.stdout to show stdout live
        # FIXME this seems quite useless.... subprocess.run seems to do exactly this
        proc = subprocess.Popen(proc_args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Retrieve std::cout and std::cerr from Threads (to raise an exception is any Matlab error happens)
        std_out_queue, std_err_queue = queue.Queue(), queue.Queue()
        self.continue_queue_threads = True
        std_out_thread = threading.Thread(target=self._enqueue_std_output, args=(proc.stdout, std_out_queue))
        std_err_thread = threading.Thread(target=self._enqueue_std_output, args=(proc.stderr, std_err_queue))
        std_out_thread.start(), std_err_thread.start()

        matlab_error_time = None
        # We keep pooling the queues until the process ends, or an error happens
        keep_running = True
        while keep_running:
            while not std_out_queue.empty():  # Does not guarantee get will not block
                try:
                    line = std_out_queue.get_nowait()
                except queue.Empty:
                    break
                self._log_and_print('[MATLAB{}] {}'.format(self._get_process_str(), line.decode('utf-8').rstrip()))
            while not std_err_queue.empty():  # Does not guarantee get will not block
                try:
                    line = std_err_queue.get_nowait()
                except queue.Empty:
                    break
                self._log_and_print('[MATLAB{} ERROR] {}'.format(self._get_process_str(),
                                                                 line.decode('utf-8').rstrip()), force_print=True)
                if matlab_error_time is None:  # Write this only once
                    matlab_error_time = datetime.now()
            time.sleep(0.001)

            if proc.poll() is not None:  # Natural ending (when script has been fully executed)
                if self.verbose:
                    print("Matlab process{} has ended by itself.".format(self._get_process_str()))
                keep_running = False
            if matlab_error_time is not None:  # Forced ending (after a small delay, to retrieve all std err data)
                if (datetime.now() - matlab_error_time).total_seconds() > 2.0:
                    keep_running = False

        if matlab_error_time is not None:
            raise RuntimeError("Matlab{} has raised an error - please check console outputs above"
                               .format(self._get_process_str()))
        rc = proc.poll()
        if rc != 0:
            warnings.warn('Matlab{} exit code was {}. Please check console outputs.'
                          .format(self._get_process_str(), rc))

        self.continue_queue_threads = False
        std_out_thread.join()
        std_err_thread.join()

        self._log_and_print("\n==================== Matlab subprocess{} has ended ========================\n"
                            .format(self._get_process_str()))

    def _log_and_print(self, log_str: str, force_print=False):
        """ Use the logger attribute if available, otherwise just print """
        if self.logger is not None:
            self.logger.log_and_print(log_str, force_print=force_print)
        elif self.verbose or force_print:
            print(log_str)


class InterpolationTimbreToolbox(ToolboxLogger):
    def __init__(self, timbre_toolbox_path: Union[str, pathlib.Path], data_root_path: Union[str, pathlib.Path],
                 verbose=True, remove_matlab_csv_after_usage=False, num_matlab_proc=1):
        """
        Can run the TimbreToolbox (in Matlab), then properly converts the computed audio features into Python
        data structures.
        :param num_matlab_proc: Number of Matlab instances created to process the entire dataset (must be >= 1).
        """
        self.timbre_toolbox_path = pathlib.Path(timbre_toolbox_path)
        self.data_root_path = pathlib.Path(data_root_path)
        self.verbose = verbose
        self._remove_csv_after_usage = remove_matlab_csv_after_usage
        self.all_sequences_features = None
        self.num_matlab_proc = num_matlab_proc
        self._logging_lock = threading.RLock()  # Re-entrant lock (with owner when acquired)

    def log_and_print(self, log_str: str, erase_file=False, force_print=False):
        with self._logging_lock:
            open_mode = 'w' if erase_file else 'a'
            with open(self.data_root_path.joinpath('timbre_matlab_log.txt'), open_mode) as f:
                f.write(log_str + '\n')  # Trailing spaces and newlines should have been removed
            if self.verbose or force_print:
                print(log_str)

    def get_audio_sub_folders(self):
        """ :returns: The List of sub-folders of the data_root_path folder. """
        return [f for f in self.data_root_path.iterdir() if f.is_dir()]

    def _clean_folders(self):
        """
        Removes all .csv and .mat files (written by TimbreToolbox) from sub-folders of the data_root_path folder.
        TODO Also remove list of folders to be analyzed by each Matlab subprocess.
        """
        for sub_dir in self.get_audio_sub_folders():
            files_to_remove = list(sub_dir.glob('*.csv'))
            files_to_remove += list(sub_dir.glob('*.mat'))
            for f in files_to_remove:
                f.unlink(missing_ok=False)
        for f in list(self.data_root_path.glob("*_input_args.txt")):
            f.unlink(missing_ok=False)

    def _get_directories_list_file(self, proc_index: int):
        """ Returns the path to the file that contains the folders that a Matlab instance will analyze. """
        return self.data_root_path.joinpath('matlab{}_input_args.txt'.format(proc_index))

    def run(self):
        t_start = datetime.now()
        self.log_and_print("Deleting all previously written .csv and .mat files...")
        self._clean_folders()

        # Get indices of sequences, using the existing folder structure only
        audio_sub_folders = sorted(self.get_audio_sub_folders())  # Are supposed to be interpolation sequence names
        sequence_names = [sub_dir.name for sub_dir in audio_sub_folders]
        try:
            sequence_indices = [int(name) for name in sequence_names]
        except ValueError:
            raise NameError("All sub-folders inside '{}' must be named as integers (indices of interpolation sequences)"
                            .format(self.data_root_path))
        if sequence_indices != list(range(len(sequence_indices))):
            raise NameError("Sub-folders inside '{}' are not named as a continuous range of indices. Found indices: {}"
                            .format(self.data_root_path, sequence_indices))

        # split list of audio folders, store them into files
        split_audio_sub_folders = np.array_split(audio_sub_folders, self.num_matlab_proc)
        for proc_index, proc_audio_folders in enumerate(split_audio_sub_folders):
            with open(self._get_directories_list_file(proc_index), 'w') as f:
                for audio_dir in proc_audio_folders:
                    f.write(str(audio_dir) + '\n')
        # run all processors in their own thread
        threads = [threading.Thread(target=self._run_matlab_proc, args=(i, self._get_directories_list_file(i)))
                   for i in range(self.num_matlab_proc)]
        for t in threads:
            t.start()
            time.sleep(1.0)  # delay thread start in order to prevent mixed console outputs
        for t in threads:
            t.join()

        # Read CSVs from all interp sequences and all audio files
        self.all_sequences_features = list()
        num_interp_steps = None
        for sub_dir in audio_sub_folders:
            if int(sub_dir.name) == 0:
                pass  # TODO sequence zero: retrieve units of all descriptors

            sequence_descriptors = list()
            csv_files = sorted([f for f in sub_dir.glob('*.csv')])
            if num_interp_steps is None:
                num_interp_steps = len(csv_files)
            elif len(csv_files) != num_interp_steps:
                raise RuntimeError("Inconsistent number of interpolation steps: found {} in {} but {} in the "
                                   "previous folder".format(len(csv_files), sub_dir, num_interp_steps))

            for step_index, csv_file in enumerate(csv_files):
                csv_index = int(csv_file.name.replace('audio_step', '').replace('_stats.csv', ''))
                if step_index != csv_index:
                    raise ValueError("Inconsistent indices. Expected step index: {}; found: {} in {}"
                                     .format(step_index, csv_index, csv_file))
                sequence_descriptors.append(self.read_stats_csv(csv_file))  # might append None (very rare)
            # Aggregate per-file descriptors and stats, into a per-sequence dict
            aggregated_seq_data = dict()
            aggregated_seq_data['step_index'] = list(range(len(sequence_descriptors)))
            # Retrieve descriptors' name - very rarely, some (zero-only) audio files can't be processed at all
            all_descriptors_names = None
            for audio_file_data in sequence_descriptors:
                if audio_file_data is not None:
                    all_descriptors_names = audio_file_data.keys()
                    break
            for descriptor_name in all_descriptors_names:
                # use zero-only descriptors values is TT could not compute actual values
                aggregated_seq_data[descriptor_name] = [
                    audio_file_data[descriptor_name] if audio_file_data is not None else 0.0
                    for audio_file_data in sequence_descriptors
                ]
            # convert data for each sequence to DF
            aggregated_seq_data = pd.DataFrame(aggregated_seq_data)
            with open(sub_dir.joinpath('tt_sequence_features.pkl'), 'wb') as f:
                pickle.dump(aggregated_seq_data, f)
            self.all_sequences_features.append(aggregated_seq_data)
        # Easy-to-load list with features from all sequences
        with open(self.data_root_path.joinpath('all_tt_sequence_features.pkl'), 'wb') as f:
            pickle.dump(self.all_sequences_features, f)

        if self._remove_csv_after_usage:
            self._clean_folders()
        delta_t = (datetime.now() - t_start).total_seconds()
        print("[InterpolationTimbreToolbox] Processed {} interpolation sequences in {:.1f} minutes ({:.2f}s/sequence)"
              .format(len(audio_sub_folders), delta_t/60.0, delta_t/len(audio_sub_folders)))

    def _run_matlab_proc(self, proc_index: int, directories_list_file: pathlib.Path):
        timbre_processor = TimbreToolboxProcess(
            self.timbre_toolbox_path, directories_list_file, verbose=self.verbose, logger=self, process_index=proc_index
        )
        timbre_processor.run()

    @staticmethod
    def _to_float(value: str, csv_file: Optional[pathlib.Path] = None, csv_line: Optional[int] = None):
        try:  # detection of casting errors
            float_value = float(value)
        except ValueError as e:
            if csv_line is not None and csv_line is not None:
                warnings.warn("Cannot convert {} to float, will use complex module instead. File '{}' "
                              "line {}: {}".format(value, csv_file, csv_line, e))
            else:
                warnings.warn("Cannot convert {} to float, will use complex module instead. {}".format(value, e))
            float_value = np.abs(complex(value.replace('i', 'j')))  # Matlab uses 'i', Python uses 'j'
        return float_value

    # ------------------------------- Features post-processing ----------------------------------

    @staticmethod  # Static
    def post_process_features(data_root_path: Union[str, pathlib.Path], freq_eps=1.0):
        """

        :param freq_eps: epsilon to be added to all features in Hz, before converting them to a log scale
        :return:
        """
        data_root_path = pathlib.Path(data_root_path)
        # 0) remove some entire columns
        seqs_features_dfs = InterpolationTimbreToolbox.get_stored_sequences_descriptors(data_root_path)
        if False:   # FIXME re-use somewhere else?
            ignored_descriptors = ('FrameErg_min', 'HarmDev_min', 'HarmErg_min', 'InHarm_min',  # mostly zeros
                                   'Noisiness'  # mostly ones
                                   'SpecSlope_min', 'SpecSlope_max', 'SpecSlope_med', 'SpecSlope_IQR',  # approx. e-7
                                   )
            for seq_idx, seq_df in enumerate(seqs_features_dfs):
                seqs_features_dfs[seq_idx] = seq_df.drop(columns=list(ignored_descriptors))
        cols = seqs_features_dfs[0].columns

        # 1) apply appropriate distortions (e.g. Hz to log scale), clamp some values (e.g. OddEvenRatio can be >> e+6)
        for seq_idx, seq_df in enumerate(seqs_features_dfs):
            for col in cols:
                # If F0 couldn't be estimated, it was set to 0 by Timbre Toolbox
                if col.startswith('F0_'):
                    seq_df[col] = np.log(freq_eps + seq_df[col])
                # Spectral Centroid, Spread, and Rolloff in Hz
                elif col.startswith('SpecCent_') or col.startswith('SpecSpread_') or col.startswith('SpecRollOff_'):
                    seq_df[col] = np.log(freq_eps + seq_df[col])
                # OddEvenRatio values explode and seem to be distributed on an exponential scale
                elif col.startswith('OddEvenRatio_'):
                    seq_df[col] = np.log(1.0 + seq_df[col])
                # Amplitudes are not scaled: FrameErg, RMS, ...

        # 2) Build arrays of aggregated data
        start_end_aggregated_features = list()
        for seq_df in seqs_features_dfs:
            start_end_aggregated_features.append(seq_df.loc[seq_df['step_index'] == 0].drop(columns=['step_index']))
            start_end_aggregated_features.append(
                seq_df.loc[seq_df['step_index'] == len(seq_df) - 1].drop(columns=['step_index']))
        start_end_aggregated_features = pd.concat(start_end_aggregated_features, ignore_index=True)

        # 3) Compute mean and std of start and end points
        # (we don't use them yet to rescale features, because we'll use the same scale across different models)
        features_stats = {
            'mean': start_end_aggregated_features.mean(),
            'std': start_end_aggregated_features.std()
        }

        # 4) TODO use the same scaling for all features

        # 5) Store processed features
        with open(data_root_path.joinpath('all_tt_postproc_sequence_features.pkl'), 'wb') as f:
            pickle.dump(seqs_features_dfs, f)
        with open(data_root_path.joinpath('tt_postproc_features_stats.pkl'), 'wb') as f:
            pickle.dump(features_stats, f)

    # ------------------------------- Read/reload stored files ----------------------------------

    def read_stats_csv(self, csv_file: pathlib.Path):
        """
        :return: A Dict of descriptors, or None if the Evaluation could not be performed by TimbreToolbox
        """
        descr_data = dict()
        with open(csv_file, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
            # if lines if a single-item array, the CSV is supposed to correspond to an "Evaluation Error" case
            if len(lines) == 1 or (len(lines) == 2 and len(lines[1]) <= 2):
                if lines[0] == 'Evaluation Error':
                    warnings.warn("File {} contains 'Evaluation Error' - the Matlab script could not "
                                  "evaluate the associated audio file".format(csv_file))
                    return None
                else:
                    raise ValueError("{} contains only 1 line: '{}' (should be audio features, or 'Evaluation Error')"
                                     .format(csv_file, lines[0]))
            # Otherwise, usual case: CSVs written by TimbreToolbox are not tabular data, but 1-item-per-line CSVs
            current_repr, current_descr = None, None
            for i, line in enumerate(lines):
                if len(line) > 0:  # Non-empty line
                    try:
                        name, value = line.split(',')
                    except ValueError:
                        continue  # e.g. empty Unit, or STFT has oddly formatted data (single line with 'Minimums,')
                    # First: we look for a new descriptor or representation
                    if name.lower() == 'representation':  # New representation: next fields are related to the
                        current_repr = value              # representation, not one of its descriptors
                        current_descr = None
                    elif name.lower() == 'descriptor':  # Next fields will be related to this descriptor
                        current_descr = value
                    else:  # At this point, the line contains a specific field
                        if current_descr is not None:  # Representation fields (min/max/med/iqr + other fields, params?)
                            if name.lower() == 'value':  # Global descriptors: single value
                                descr_data[current_descr] = self._to_float(value, csv_file, i)
                            elif name.lower() == 'minimum':
                                descr_data[current_descr + '_min'] = self._to_float(value, csv_file, i)
                            elif name.lower() == 'maximum':
                                descr_data[current_descr + '_max'] = self._to_float(value, csv_file, i)
                            elif name.lower() == 'median':
                                descr_data[current_descr + '_med'] = self._to_float(value, csv_file, i)
                            elif name.lower() == 'interquartile range':
                                descr_data[current_descr + '_IQR'] = self._to_float(value, csv_file, i)
                            else:  # Other descriptor fields (e.g. unit, parameters) are discarded
                                pass
                else:  # Empty line indicates a new descriptor (maybe a new representation)
                    current_descr = None
        return descr_data

    @staticmethod
    def get_stored_sequences_descriptors(data_root_path: Union[str, pathlib.Path]):
        with open(pathlib.Path(data_root_path).joinpath('all_tt_sequence_features.pkl'), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_stored_postproc_sequences_descriptors(data_root_path: Union[str, pathlib.Path]):
        with open(pathlib.Path(data_root_path).joinpath('all_tt_postproc_sequence_features.pkl'), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_stored_postproc_descriptors_stats(data_root_path: Union[str, pathlib.Path]):
        with open(pathlib.Path(data_root_path).joinpath('tt_postproc_features_stats.pkl'), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_default_postproc_features_stats():
        """
        Stats from InterpNaive/interp_validation than can be used to rescale all descriptors,
         for numerical stability.
        Using these on any dataset does not lead to data leakage, because these features are used for the
         evaluation only (not for training), and we'll measure variations between feature values for different
         models (so the scale does not matter.
        """
        d = {
            'mean': {
                'FrameErg_min': 8.010944684743496e-05, 'FrameErg_max': 0.08964752405643552,
                'FrameErg_med': 0.02425811735224146, 'FrameErg_IQR': 0.023518792322612732,
                'SpecCent_min': 3.571549663409616, 'SpecCent_max': 6.962772362103871,
                'SpecCent_med': 5.792311844913885, 'SpecCent_IQR': 4.376653138775212,
                'SpecCrest_min': 27.908622992574237, 'SpecCrest_max': 194.9429661056109,
                'SpecCrest_med': 92.84861313531319, 'SpecCrest_IQR': 30.60170374296378,
                'SpecDecr_min': -3.376656950521448, 'SpecDecr_max': 0.0977758678877888,
                'SpecDecr_med': -0.03573709104620463, 'SpecDecr_IQR': 0.15491952879646864,
                'SpecFlat_min': 0.0006274630453597367, 'SpecFlat_max': 0.20514838185808548,
                'SpecFlat_med': 0.01969614086468646, 'SpecFlat_IQR': 0.02798541127906598,
                'SpecKurt_min': 9.693788481848186, 'SpecKurt_max': 17462.422784158403,
                'SpecKurt_med': 1417.3470056105605, 'SpecKurt_IQR': 1762.0204606056143,
                'SpecRollOff_min': 4.47369232351795, 'SpecRollOff_max': 7.9093891169182475,
                'SpecRollOff_med': 6.431576808404387, 'SpecRollOff_IQR': 4.84138119717418,
                'SpecSkew_min': 0.38533883203630354, 'SpecSkew_max': 94.650225259076,
                'SpecSkew_med': 12.538739981320116, 'SpecSkew_IQR': 11.918588675775585,
                'SpecSlope_min': -1.4181556204620651e-06, 'SpecSlope_max': -8.786587823762383e-07,
                'SpecSlope_med': -1.2739597323762379e-06, 'SpecSlope_IQR': 8.147403468240603e-08,
                'SpecSpread_min': 3.7761219404580033, 'SpecSpread_max': 6.909128212087304,
                'SpecSpread_med': 5.258077053595149, 'SpecSpread_IQR': 4.526939795637583,
                'SpecVar_min': 0.0006342211731345065, 'SpecVar_max': 0.704256513861386,
                'SpecVar_med': 0.06415011062030543, 'SpecVar_IQR': 0.062312537747684235, 'F0_min': 3.7208276365252795,
                'F0_max': 5.602909769960698, 'F0_med': 4.940743858811075, 'F0_IQR': 0.9299377477419094,
                'HarmDev_min': 3.768821066716206e-07, 'HarmDev_max': 0.0004729706843564364,
                'HarmDev_med': 0.00012964380989658947, 'HarmDev_IQR': 0.000155028130723446,
                'HarmErg_min': 8.037911999998512e-09, 'HarmErg_max': 0.0005887197139999287,
                'HarmErg_med': 9.874854003233112e-05, 'HarmErg_IQR': 0.00015287747905632794,
                'InHarm_min': 1.1432295585701262e-05, 'InHarm_max': 0.46528585895776575,
                'InHarm_med': 0.007002859643241572, 'InHarm_IQR': 0.011774197080586097,
                'NoiseErg_min': 0.0016273083490026138, 'NoiseErg_max': 0.7617621646336667,
                'NoiseErg_med': 0.2298782368800161, 'NoiseErg_IQR': 0.21693375653789165,
                'Noisiness_min': 0.9969886537953758, 'Noisiness_max': 0.9999999702970297,
                'Noisiness_med': 0.9998741353135288, 'Noisiness_IQR': 0.0002572869011653661,
                'OddEvenRatio_min': 0.034753347209729564, 'OddEvenRatio_max': 11.929637166552343,
                'OddEvenRatio_med': 3.6846388097691243, 'OddEvenRatio_IQR': 5.265543746259746,
                'AmpMod': 0.08289282689438927, 'Att': 0.06284253102310228, 'AttSlope': 9.587411297029698,
                'Dec': 0.36104491254125415, 'DecSlope': -1.0583661035116079, 'EffDur': 2.0506328366336626,
                'FreqMod': 3.3819141584158094, 'LAT': -0.9855746346204641, 'RMSEnv_min': 0.00020379769142079266,
                'RMSEnv_max': 0.12104371676567678, 'RMSEnv_med': 0.05500859121104683,
                'RMSEnv_IQR': 0.035394505695014865, 'Rel': 2.1642007953795415, 'TempCent': 1.2450313630363052
            },
            'std' : {
                'FrameErg_min': 0.0008269783562767417, 'FrameErg_max': 0.10267535612066643,
                'FrameErg_med': 0.0472796815171985, 'FrameErg_IQR': 0.03756240571654556,
                'SpecCent_min': 1.334086160674959, 'SpecCent_max': 0.8201302796575851,
                'SpecCent_med': 0.9382255417888526, 'SpecCent_IQR': 1.6121119007764708,
                'SpecCrest_min': 15.752138457574173, 'SpecCrest_max': 49.3718173586158,
                'SpecCrest_med': 34.17790415752547, 'SpecCrest_IQR': 24.21545343496933,
                'SpecDecr_min': 2.187233556490388, 'SpecDecr_max': 0.07031407044783261,
                'SpecDecr_med': 0.5127110938756229, 'SpecDecr_IQR': 0.5634883337900121,
                'SpecFlat_min': 0.011345457207456994, 'SpecFlat_max': 0.31065480540506757,
                'SpecFlat_med': 0.10726546104719793, 'SpecFlat_IQR': 0.12796623550377897,
                'SpecKurt_min': 32.151434866781756, 'SpecKurt_max': 11027.269482340484,
                'SpecKurt_med': 4002.8733746710795, 'SpecKurt_IQR': 4072.561471529097,
                'SpecRollOff_min': 1.1035554825368659, 'SpecRollOff_max': 0.7622243541601438,
                'SpecRollOff_med': 1.0408963879981736, 'SpecRollOff_IQR': 2.2991160956672148,
                'SpecSkew_min': 5.129337876063248, 'SpecSkew_max': 47.551185504139156,
                'SpecSkew_med': 22.55919807334764, 'SpecSkew_IQR': 19.76315953471566,
                'SpecSlope_min': 1.1740150652228536e-07, 'SpecSlope_max': 5.841834348078272e-07,
                'SpecSlope_med': 2.4934440839797783e-07, 'SpecSlope_IQR': 1.6352353845967834e-07,
                'SpecSpread_min': 0.7047388504105024, 'SpecSpread_max': 0.669245038583682,
                'SpecSpread_med': 1.022293984849524, 'SpecSpread_IQR': 1.3642415395849556,
                'SpecVar_min': 0.006398578370536717, 'SpecVar_max': 0.37717106102347747,
                'SpecVar_med': 0.2254046619839119, 'SpecVar_IQR': 0.20477677612724454, 'F0_min': 0.7584213180127082,
                'F0_max': 1.1677579615483809, 'F0_med': 1.0938805814331705, 'F0_IQR': 1.368153267132658,
                'HarmDev_min': 4.754507852346634e-06, 'HarmDev_max': 0.0005901919372487966,
                'HarmDev_med': 0.0002924299126316289, 'HarmDev_IQR': 0.000286398402917251,
                'HarmErg_min': 1.9068393284541372e-07, 'HarmErg_max': 0.003405115051446283,
                'HarmErg_med': 0.0009724165860547822, 'HarmErg_IQR': 0.0010889683446972979,
                'InHarm_min': 0.00022745586680282542, 'InHarm_max': 1.1479130346791733,
                'InHarm_med': 0.13440462316164742, 'InHarm_IQR': 0.1129042680635482,
                'NoiseErg_min': 0.017268031435842077, 'NoiseErg_max': 0.879069769880698,
                'NoiseErg_med': 0.44661241729042506, 'NoiseErg_IQR': 0.34182643316319233,
                'Noisiness_min': 0.02732977124527132, 'Noisiness_max': 6.54447852496754e-07,
                'Noisiness_med': 0.0008950247805038345, 'Noisiness_IQR': 0.002409596730127832,
                'OddEvenRatio_min': 0.41869429859942137, 'OddEvenRatio_max': 7.192133075843989,
                'OddEvenRatio_med': 5.77464025677899, 'OddEvenRatio_IQR': 6.437053360836346,
                'AmpMod': 0.11146779857719995, 'Att': 0.13075001940841233, 'AttSlope': 2.862875422418556,
                'Dec': 0.4624598854938087, 'DecSlope': 2.119534086253417, 'EffDur': 1.1355132121831883,
                'FreqMod': 1.8065808763903015, 'LAT': 0.24414176395310339, 'RMSEnv_min': 0.0004869751241989705,
                'RMSEnv_max': 0.06290776100103077, 'RMSEnv_med': 0.050936677287260895,
                'RMSEnv_IQR': 0.028225859576089866, 'Rel': 1.168398160570344, 'TempCent': 0.5174519131911464
            }
        }
        return d


if __name__ == "__main__":

    _timbre_toolbox_path = '~/Documents/MATLAB/timbretoolbox'
    _root_path = pathlib.Path(__file__).resolve().parent.parent
    _data_root_path = '/home/gwendal/Jupyter/Data_SSD/Interpolations/LinearNaive/interp_validation'  # TODO anonymize

    timbre_proc = InterpolationTimbreToolbox(_timbre_toolbox_path, _data_root_path,
                                             num_matlab_proc=12)  # 12 for bigger datasets
    timbre_proc.run()

    # TODO REMOVE, debug temp
    InterpolationTimbreToolbox.post_process_features(_data_root_path)

