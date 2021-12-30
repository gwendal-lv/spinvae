"""
Functions and classes to easily load models or data for evaluation.
"""

from pathlib import Path

import torch.cuda

import data.build
import logs.logger
import model.build
import utils.config


class ModelLoader:
    def __init__(self, model_folder_path: str, device='cpu', dataset_type='validation'):
        """
        Loads model_config and train_config from the given folder, then builds the model and loads the last
        available checkpoint.
        The dataset as well as the required dataloader are also built.

        :param model_folder_path: relative to the project's root folder. E.g. : saved/MMD_tests/model1
        :param dataset_type: Usually 'validation' or 'test', but 'train' is accepted.
        """
        self._root_path = Path(__file__).resolve().parent.parent
        self.device = device
        self.path_to_model_dir = self._root_path.joinpath(model_folder_path)
        self.model_config, self.train_config = utils.config.get_config_from_file(
            self.path_to_model_dir.joinpath('config.json'))
        if device == 'cpu':
            self.train_config.main_cuda_device_idx = -1

        checkpoint = logs.logger.get_model_last_checkpoint(self._root_path, self.model_config, device=self.device)
        if self.train_config.pretrain_ae_only:
            # TODO load dataset + dataloader
            self.dataset = None
            self.dataset_type = 'None'
            # Load model
            _, _, self.ae_model = model.build.build_ae_model(self.model_config, self.train_config)
            self.ae_model.load_checkpoint(checkpoint, eval_only=True)
            self.reg_model, self.extended_ae_model = None, None
        else:
            # Dataset required to build the preset indexes helper
            self.dataset = data.build.get_dataset(self.model_config, self.train_config)
            self.dataset_type = dataset_type
            dataloaders, dataloaders_nb_items \
                = data.build.get_split_dataloaders(self.train_config, self.dataset, num_workers=0)
            self.dataloader, self.dataloader_num_items = dataloaders[dataset_type], dataloaders_nb_items[dataset_type]
            # Load model
            _, _, self.ae_model, self.reg_model, self.extended_ae_model = model.build.build_extended_ae_model(
                self.model_config, self.train_config, self.dataset.preset_indexes_helper)
            self.ae_model.load_checkpoint(checkpoint, eval_only=True)
            self.reg_model.load_checkpoint(checkpoint, eval_only=True)
        del checkpoint
        if device == 'cpu':
            torch.cuda.empty_cache()  # Checkpoints were usually GPU tensors (originally)


if __name__ == "__main__":
    loader = ModelLoader('saved/ControlsRegr_allascat/htanhTrue_softmT°0.2_permTrue')
    print("OK")

