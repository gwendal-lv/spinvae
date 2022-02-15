
import subprocess
import pathlib
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

    def run(self):
        # Matlab args From https://arc.umich.edu/software/matlab/
        # and https://stackoverflow.com/questions/38723138/matlab-execute-script-from-command-linux-line/38723505
        proc_args = ['matlab', '-nodisplay', '-nodesktop', '-nosplash', '-r', self.matlab_commands]
        log_str = '============ Launching matlab commands (will block if a Matlab error happens) ============\n' \
                  '{}\n' \
                  'Subprocess args: {}\n'.format(datetime.now().strftime("%Y/%m/%d, %H:%M:%S"), proc_args)
        self._log_and_print(log_str, erase_file=True)

        # Subprocess: use a single string if shell is True, otherwise use a list of command and args
        # TODO
        #  - time out? dur de prévoir combien un calcul complet va prendre...
        #  - récupérer std cout en live ?
        proc_results = subprocess.run(proc_args, shell=False, capture_output=True)
        stdout_str = proc_results.stdout.decode('utf-8')

        log_str = "================== Matlab (in Python-subprocess) std out: ==================\n"
        log_str += stdout_str
        log_str += "\n=============== Matlab (in Python-subprocess) std err: ==================\n"
        log_str += proc_results.stderr.decode('utf-8')
        log_str += "\n==================== Matlab subprocess has ended ========================\n"
        self._log_and_print(log_str)

        # TODO check for errors - display std cout if not verbose
        if len(proc_results.stderr) > 0:
            raise RuntimeError("Matlab std::err: \n{}".format(proc_results.stderr.decode('utf-8')))
        elif False:
            pass  #TODO
        rien = 0

    def _log_and_print(self, log_str, erase_file=False):
        open_mode = 'w' if erase_file else 'a'
        with open(self.data_root_path.joinpath('timbre_matlab_log.txt'), open_mode) as f:
            f.write(log_str)
        if self.verbose:
            print(log_str)


if __name__ == "__main__":

    _timbre_toolbox_path = '/home/gwendal/Documents/MATLAB/timbretoolbox'
    _data_root_path = '/media/gwendal/Data/Interpolations/LinearNaive/interp_validation'
    timbre_proc = TimbreToolboxProcess(_timbre_toolbox_path, _data_root_path)
    timbre_proc.run()


