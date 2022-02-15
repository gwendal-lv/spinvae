import queue
import subprocess
import pathlib
import threading
import time
import warnings
from datetime import datetime


class TimbreToolboxProcess:
    def __init__(self, timbre_toolbox_path: str, data_root_path: str, verbose=True):
        """
        Runs the TimbreToolbox.
        The 'matlab' command must be available system-wide.

        :param timbre_toolbox_path: Path the TimbreToolbox https://github.com/VincentPerreault0/timbretoolbox
        :param data_root_path: Folder whose sub-folders contain audio samples to be analyzed.
        :param verbose:
        """
        self.timbre_toolbox_path = timbre_toolbox_path
        self.data_root_path = pathlib.Path(data_root_path)
        self.verbose = verbose
        self.current_path = str(pathlib.Path(__file__).parent)
        self.matlab_commands = "addpath(genpath('{}')); " \
                               "cd '{}'; " \
                               "timbre; " \
                               "exit " \
            .format(self.timbre_toolbox_path,  # Path to the Timbre Toolbox
                    self.current_path,  # Path to the local timbre.m file
                    )
        if verbose:
            print("Current path for the timbre.m Matlab function: {}".format(self.current_path))

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
        self._log_and_print(log_str, erase_file=True)

        # Poll process.stdout to show stdout live
        proc = subprocess.Popen(proc_args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        matlab_error_time = None

        # Retrieve std::cout and std::cerr from Threads (to raise an exception is any Matlab error happens)
        std_out_queue, std_err_queue = queue.Queue(), queue.Queue()
        self.continue_queue_threads = True
        std_out_thread = threading.Thread(target=self._enqueue_std_output, args=(proc.stdout, std_out_queue))
        std_err_thread = threading.Thread(target=self._enqueue_std_output, args=(proc.stderr, std_err_queue))
        std_out_thread.start(), std_err_thread.start()

        matlab_error_time = None
        while True:
            while not std_out_queue.empty():  # Does not guarantee get will not block
                try:
                    line = std_out_queue.get_nowait()
                except queue.Empty:
                    break
                self._log_and_print('[MATLAB] {}'.format(line.decode('utf-8').rstrip()))
            while not std_err_queue.empty():  # Does not guarantee get will not block
                try:
                    line = std_err_queue.get_nowait()
                except queue.Empty:
                    break
                self._log_and_print('[MATLAB ERROR] {}'.format(line.decode('utf-8').rstrip()), force_print=True)
                if matlab_error_time is None:  # Write this only once
                    matlab_error_time = datetime.now()
            time.sleep(0.001)

            if proc.poll() is not None:  # Natural ending (when script has been fully executed)
                if self.verbose:
                    print("Matlab process has ended by itself.")
                break
            if matlab_error_time is not None:  # Forced ending (after a small delay, to retrieve all std err data)
                if (datetime.now() - matlab_error_time).total_seconds() > 2.0:
                    break

        if matlab_error_time is not None:
            # TODO display cerr data, if not verbose
            raise RuntimeError("Matlab has raised an error - please check console outputs above")
        rc = proc.poll()
        if rc != 0:
            warnings.warn('Matlab exit code was {}. Please check console outputs.'.format(rc))

        self.continue_queue_threads = False
        std_out_thread.join()
        std_err_thread.join()

        self._log_and_print("\n==================== Matlab subprocess has ended ========================\n")

    def _log_and_print(self, log_str, erase_file=False, force_print=False):
        open_mode = 'w' if erase_file else 'a'
        with open(self.data_root_path.joinpath('timbre_matlab_log.txt'), open_mode) as f:
            f.writelines([log_str])
        if self.verbose or force_print:
            print(log_str)


if __name__ == "__main__":

    _timbre_toolbox_path = '/home/gwendal/Documents/MATLAB/timbretoolbox'
    _data_root_path = '/media/gwendal/Data/Interpolations/LinearNaive/interp_validation'
    timbre_proc = TimbreToolboxProcess(_timbre_toolbox_path, _data_root_path)
    timbre_proc.run()


