"""
Functions and classes to easily load models or data for evaluation.
"""

from pathlib import Path

import data.build
import logs.logger
import model.build
import utils.config


class ModelLoader:
    def __init__(self, model_folder_path: str, device='cpu'):
        """
        Loads model_config and train_config from the given folder, then builds the model and loads the last
        available checkpoint.

        :param model_folder_path: relative to the project's root folder. E.g. : saved/MMD_tests/model1
        """
        self._root_path = Path(__file__).resolve().parent.parent
        self.device = device
        self.path_to_model_dir = self._root_path.joinpath(model_folder_path)
        self.model_config, self.train_config = utils.config.get_config_from_file(self.path_to_model_dir
                                                                                 .joinpath('config.json'))
        checkpoint = logs.logger.get_model_last_checkpoint(self._root_path, self.model_config, device=self.device)
        if self.train_config.pretrain_ae_only:
            _, _, self.ae_model = model.build.build_ae_model(self.model_config, self.train_config)
            self.ae_model.load_checkpoint(checkpoint, eval_only=True)
            self.reg_model, self.extended_ae_model = None, None
            # TODO load dataset(s)
            self.dataset = None
        else:
            # Dataset required to build the preset indexes helper
            self.dataset = data.build.get_dataset(self.model_config, self.train_config)
            _, _, self.ae_model, self.reg_model, self.extended_ae_model = model.build.build_extended_ae_model(
                self.model_config, self.train_config, self.dataset.preset_indexes_helper)
            self.ae_model.load_checkpoint(checkpoint, eval_only=True)
            self.reg_model.load_checkpoint(checkpoint, eval_only=True)


if __name__ == "__main__":
    loader = ModelLoader('saved/ControlsRegr_allascat/htanhTrue_softmT°0.2_permTrue')
    print("OK")

