# renamed to profiling.py (instead of profile.py) - otherwise, name clash with seaborn

import contextlib
from pathlib import Path

import torch.profiler



class OptionalProfiler:
    def __init__(self, train_config, tensorboard_run_dir: Path):
        self.enabled = train_config.profiler_enabled
        self.verbosity = train_config.verbosity
        self.prof_kwargs = train_config.profiler_kwargs
        self.prof_sched_kwargs = train_config.profiler_schedule_kwargs
        self.epoch_to_profile = train_config.profiler_epoch_to_record
        self.tensorboard_run_dir = tensorboard_run_dir

    def get_prof(self, epoch):
        """ Might return an torch.profile.profiler, or a NoProfiler if this epoch is not to be recorded. """
        if self.enabled and epoch == self.epoch_to_profile:
            if self.verbosity > 0:
                print("Profiling epoch {}. Log dir: {}".format(epoch, self.tensorboard_run_dir))
            return torch.profiler.profile(
                schedule=(torch.profiler.schedule(**self.prof_sched_kwargs)),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.tensorboard_run_dir),
                **self.prof_kwargs
            )
        else:
            return contextlib.nullcontext()

