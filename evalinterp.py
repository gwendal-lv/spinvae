import os.path
from pathlib import Path
from typing import List, Dict, Union

from evaluation.load import ModelLoader
from evaluation.interpbase import NaivePresetInterpolation
from evaluation.interp import SynthPresetLatentInterpolation
from utils import config_confidential



class EvalConfig:
    def __init__(self):
        self.device = 'cpu'
        self.dataset_type = 'validation'
        self.num_steps = 9
        self.use_reduced_dataset = True  # fast debugging (set to False during actual eval)
        self.force_re_eval_all = False

        self.logs_root_dir = Path(config_confidential.logs_root_dir)
        # Reference data can be stored anywhere (they don't use a trained NN)
        self.reference_model_path = self.logs_root_dir.parent.joinpath('RefInterp/LinearNaive')
        self.ref_model_force_re_eval = False

        # TODO list of models and eval configs (as kwargs) for each model
        #    the config of the first model will be used to load the dataset (to be used by the reference model)
        self.other_models: List[Dict[str, Union[str, Path, bool]]] = [
            {
                'base_model_name': 'presetAE/combined_vae_beta1.60e-04_presetfactor0.50',
                'u_curve': 'linear',
                'latent_interp': 'linear',
                'force_re_eval': False
            },
            {
                'base_model_name': 'presetAE/combined_vae_beta1.60e-04_presetfactor0.20',
                'u_curve': 'linear',
                'latent_interp': 'linear',
            }
        ]

        # TODO option to force re-eval models that have already been evaluated

        self.build_models_storage_path()

    def build_models_storage_path(self):
        """ auto build eval data paths from the model name and interp-hyperparams """
        for m_config in self.other_models:
            m_config['base_model_path'] = self.logs_root_dir.joinpath(m_config['base_model_name'])
            interp_name = 'interp{}'.format(self.num_steps)
            interp_name += '_' + self.dataset_type[0:5]
            interp_name += '_u' + m_config['u_curve'][0:3].capitalize()
            interp_name += '_z' + m_config['latent_interp'][0:3].capitalize()
            m_config['interp_storage_path'] = m_config['base_model_path'].joinpath(interp_name)




def run(eval_config: EvalConfig):

    assert eval_config.dataset_type == 'validation' or eval_config.dataset_type == 'test'
    model_loader = ModelLoader(
        eval_config.other_models[0]['base_model_path'], eval_config.device, eval_config.dataset_type)

    if eval_config.reference_model_path.name == 'LinearNaive':
        ref_preset_interpolator = NaivePresetInterpolation(
            model_loader.dataset, model_loader.dataset_type, model_loader.dataloader,
            eval_config.reference_model_path, num_steps=eval_config.num_steps, u_curve='linear'
        )
    else:
        raise NotImplementedError()
    ref_preset_interpolator.use_reduced_dataset = eval_config.use_reduced_dataset
    ref_preset_interpolator.try_process_dataset(eval_config.ref_model_force_re_eval or eval_config.force_re_eval_all)

    # Evaluate all "other" (i.e. non-reference) models
    for m_config in eval_config.other_models:
        print("\n\n\n\n--------------------------------------------------------------------------------------------")
        model_loader = ModelLoader(m_config['base_model_path'], eval_config.device, eval_config.dataset_type)
        preset_interpolator = SynthPresetLatentInterpolation(
            model_loader, storage_path=m_config['interp_storage_path'],
            num_steps=eval_config.num_steps, u_curve=m_config['u_curve'], latent_interp=m_config['latent_interp']
        )
        preset_interpolator.use_reduced_dataset = eval_config.use_reduced_dataset
        try:
            force_re_eval = m_config['force_re_eval']
        except KeyError:  # Default: don't force re-evaluate
            force_re_eval = False
        force_re_eval = force_re_eval or eval_config.force_re_eval_all
        preset_interpolator.try_process_dataset(force_re_eval)

    # TODO build figureS: all models compared together, all compared to ref (one by one)


if __name__ == "__main__":
    run(EvalConfig())
