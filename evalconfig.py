"""
Allows easy modification of all configuration parameters required to perform a series of models evaluations.
This script is not intended to be run, it only describes parameters.
"""

from pathlib import Path
from typing import List, Dict, Union

from utils import config_confidential


class InterpEvalConfig:
    def __init__(self):
        self.device = 'cpu'
        self.dataset_type = 'validation'
        self.num_steps = 9
        self.use_reduced_dataset = True  # fast debugging (set to False during actual eval)
        self.force_re_eval_all = False
        self.skip_audio_render = False  # don't re-render audio, recompute interpolation features/metrics only

        # Audio features and interpolation metrics
        self.exclude_min_max_interp_features = True  # the TimbreToolbox paper advises to use IQR and medians only
        # Features to be rejected
        #    - Noisiness features seem badly estimated for the DX7 (because of the FM?). They are quite constant
        #      equal to 1.0 (absurd) and any slightly < 1.0 leads to diverging values after 1-std normalization
        self.excluded_interp_features = ('Noisiness', )

        # Reference model
        self.logs_root_dir = Path(config_confidential.logs_root_dir)
        # Reference data can be stored anywhere (they don't use a trained NN)
        self.reference_model_path = self.logs_root_dir.parent.joinpath('RefInterp/LinearNaive')
        self.ref_model_interp_path = self.reference_model_path.joinpath(
            'interp{}_{}'.format(self.num_steps, self.dataset_type[0:5]))
        self.ref_model_force_re_eval = False

        # TODO try different u/z curves
        # List of models and eval configs for each model
        #    the config of the first model will be used to load the dataset used by the reference model
        self.other_models: List[Dict[str, Union[str, Path, bool]]] = [
            {'base_model_name': 'presetAE/combined_vae_beta1.6e-03_presetfactor1.00',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.6e-03_presetfactor0.50',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.6e-03_presetfactor0.20',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.6e-04_presetfactor1.00',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.60e-04_presetfactor0.50',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.60e-04_presetfactor0.20',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.6e-05_presetfactor1.00',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.60e-05_presetfactor0.50',
             'u_curve': 'linear', 'latent_interp': 'linear'},
            {'base_model_name': 'presetAE/combined_vae_beta1.60e-05_presetfactor0.20',
             'u_curve': 'linear', 'latent_interp': 'linear'},
        ]

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
            m_config['model_interp_name'] = m_config['base_model_name'] + '/' + interp_name
