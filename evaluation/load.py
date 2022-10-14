"""
Functions and classes to easily load models or data for evaluation.
"""

from pathlib import Path
import pickle

import torch.cuda

import config
import data.build
import logs.logger
import model.hierarchicalvae


class ModelLoader:
    def __init__(self, model_folder_path: Path, device='cpu', dataset_type='validation'):
        """
        Loads model_config and train_config from the given folder, then builds the model and loads the last
        available checkpoint.
        The dataset as well as the required dataloader are also built.

        :param model_folder_path: e.g. Path('/home/user/saved/MMD_tests/model1')
        :param dataset_type: Usually 'validation' or 'test', but 'train' is accepted.
        """
        self.device = device
        self.path_to_model_dir = model_folder_path
        self.model_config, self.train_config = self.get_model_train_configs(self.path_to_model_dir)
        if device == 'cpu':
            self.train_config.main_cuda_device_idx = -1

        # Build dataset, dataloaders
        self.dataset_type = dataset_type
        if self.train_config.pretrain_audio_only:
            raise NotImplementedError()
        else:
            # Parts of train.py code
            self.dataset = data.build.get_dataset(self.model_config, self.train_config)
            dataloaders, dataloaders_nb_items = data.build.get_split_dataloaders(self.train_config, self.dataset)
            self.dataloader = dataloaders[self.dataset_type]
            self.dataloader_num_items = dataloaders_nb_items[self.dataset_type]

        # Then build model and load its weights
        self.model_config.dim_z = -1  # Will be set by the hVAE itself
        self.ae_model = model.hierarchicalvae.HierarchicalVAE(
            self.model_config, self.train_config, self.dataset.preset_indexes_helper)
        self.ae_model.load_checkpoints(self.path_to_model_dir.joinpath("checkpoint.tar"))  # FIXME specify device

        if device == 'cpu':
            torch.cuda.empty_cache()  # Checkpoints were usually GPU tensors (originally)

    @staticmethod
    def get_model_train_configs(model_dir: Path):
        with open(model_dir.joinpath("config.pickle"), 'rb') as f:
            checkpoint_configs = pickle.load(f)
        model_config: config.ModelConfig = checkpoint_configs['model']
        train_config: config.TrainConfig = checkpoint_configs['train']
        return model_config, train_config


if __name__ == "__main__":
    _model_path = Path(__file__).resolve().parent.parent
    _model_path = _model_path.joinpath('../Data_SSD/Logs/preset-vae/dev/presetAE_tfm_ff_00')
    loader = ModelLoader(_model_path)
    print("OK")

