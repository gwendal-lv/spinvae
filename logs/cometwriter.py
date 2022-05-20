
from comet_ml import Experiment


class CometWriter:
    def __init__(self, model_config, train_config):  # TODO typing hint after config.py refactoring
        # We'll create a new experiment, or load an existing one if restarting from a checkpoint
        if train_config.start_epoch == 0:
            self.experiment = Experiment(
                api_key=model_config.comet_api_key, workspace=model_config.comet_workspace,
                project_name=model_config.comet_project_name,
                log_git_patch=False,  # Quite large, and useless during remote dev (different commit on remote machine)
            )
            self.experiment.set_name(model_config.name + '/' + model_config.run_name)
        else:
            raise NotImplementedError()  # TODO implement ExistingExperiment
