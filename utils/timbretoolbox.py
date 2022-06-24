import multiprocessing.pool
import pickle
import queue
import subprocess
import pathlib
import threading
import time
import warnings
from datetime import datetime
from typing import Optional
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
    def __init__(self, timbre_toolbox_path: str, data_root_path: str, verbose=True,
                 remove_matlab_csv_after_usage=False,  # FIXME default to true (or auto set to False during debug)
                 num_matlab_proc=1):
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
                sequence_descriptors.append(self.read_stats_csv(csv_file))
            # Aggregate per-file descriptors and stats, into a per-sequence dict
            aggregated_seq_data = dict()
            aggregated_seq_data['seq_index'] = list(range(len(sequence_descriptors)))
            for descriptor_name in sequence_descriptors[0].keys():
                aggregated_seq_data[descriptor_name] \
                    = [audio_file_data[descriptor_name] for audio_file_data in sequence_descriptors]
            # convert data for each sequence to DF
            aggregated_seq_data = pd.DataFrame(aggregated_seq_data)
            with open(sub_dir.joinpath('sequence_features.pkl'), 'wb') as f:
                pickle.dump(aggregated_seq_data, f)
            self.all_sequences_features.append(aggregated_seq_data)
        # Easy-to-load list with features from all sequences
        with open(self.data_root_path.joinpath('all_sequence_features.pkl'), 'wb') as f:
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

    def read_stats_csv(self, csv_file: pathlib.Path):
        """
        :return: A Dict of descriptors. Some of them
        """
        descr_data = dict()
        with open(csv_file, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
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
    def get_stored_sequences_descriptors(data_root_path: str):
        with open(pathlib.Path(data_root_path).joinpath('all_sequence_features.pkl'), 'rb') as f:
            return pickle.load(f)



if __name__ == "__main__":

    _timbre_toolbox_path = '/home/gwendal/Documents/MATLAB/timbretoolbox'
    _data_root_path = '/media/gwendal/Data/Interpolations/LinearNaive/interp_validation'
    timbre_proc = InterpolationTimbreToolbox(_timbre_toolbox_path, _data_root_path,
                                             num_matlab_proc=12)
    timbre_proc.run()


