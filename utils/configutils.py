
import json
import collections

import config


# Could (should) be improved...
class _Config(object):
    pass

# FIXME move directly to ../evalconfig.py
class EvalConfig:
    def __init__(self):
        self.start_datetime = ''
        self.models_names = ['']
        self.override_previous_eval = False
        self.device = ''
        self.verbosity = 0
        self.dataset = ''
        self.k_folds_count = 0
        self.minibatch_size = 0
        self.load_from_archives = False
        self.multiprocess_cores_ratio = 0.5


