
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from .metrics import BufferedMetric, LatentMetric

import utils.stat


class CorrectedSummaryWriter(SummaryWriter):
    """ SummaryWriter corrected to prevent extra runs to be created
    in Tensorboard when adding hparams.

    Original code in torch/utils/tensorboard.writer.py,
    modification by method overloading inspired by https://github.com/pytorch/pytorch/issues/32651 """

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        assert run_name is None  # Disabled feature. Run name init by summary writer ctor

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        # run_name argument is discarded and the writer itself is used (no extra writer instantiation)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


class TensorboardSummaryWriter(CorrectedSummaryWriter):
    """ Tensorboard SummaryWriter with corrected add_hparams method
     and extra functionalities. """

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='',
                 model_config=None, train_config=None  # Added (actually mandatory) arguments
                 ):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        # Full-Config is required. Default constructor values allow to keep the same first constructor args
        self.model_config = model_config
        self.train_config = train_config
        self.resume_from_checkpoint = (train_config.start_epoch > 0)
        self.hyper_params = dict()
        self.hparams_domain_discrete = dict()  # TODO hparam domain discrete
        # General and dataset hparams
        self.hyper_params['batchsz'] = self.train_config.minibatch_size
        self.hyper_params['kfold'] = self.train_config.current_k_fold
        self.hparams_domain_discrete['kfold'] = list(range(self.train_config.k_folds))
        self.hyper_params['wdecay'] = self.train_config.weight_decay
        self.hyper_params['synth'] = self.model_config.synth
        self.hyper_params['syntargs'] = self.model_config.synth_args_str
        self.hyper_params['nmidi'] = '{}{}'.format(len(self.model_config.midi_notes),
                                                   ("stack" if model_config.stack_spectrograms else "inde"))
        self.hyper_params['catcontmodel'] = self.model_config.synth_vst_params_learned_as_categorical
        self.hyper_params['normalizeloss'] = self.train_config.normalize_losses
        # Latent space hparams
        self.hyper_params['z_dim'] = self.model_config.dim_z
        self.hyper_params['latloss'] = self.train_config.latent_loss
        self.hyper_params['latbeta'] = self.train_config.beta
        self.hyper_params['stylearch'] = self.model_config.style_architecture
        self.hyper_params['latfl_arch'] = self.model_config.latent_flow_arch
        self.hyper_params['latfl_in_regul'] = self.train_config.latent_flow_input_regularization
        self.hyper_params['mmd_n_est'] = self.train_config.mmd_num_estimates
        # Synth controls regression
        self.hyper_params['ncontrols'] = self.model_config.synth_params_count
        # self.hyper_params['contloss'] = self.model_config.controls_losses
        self.hyper_params['regr_arch'] = self.model_config.params_regression_architecture
        self.hyper_params['regr_FCdrop'] = self.train_config.reg_fc_dropout
        self.hyper_params['regr_outsoftm'] = self.model_config.params_reg_softmax
        self.hyper_params['regr_catloss'] = 'BinCE' if self.train_config.params_cat_bceloss else 'CatCE'
        # Auto-Encoder hparams
        self.hyper_params['VAE_FCdrop'] = self.train_config.fc_dropout
        self.hyper_params['enc_arch'] = self.model_config.encoder_architecture
        self.hyper_params['recons_loss'] = self.train_config.reconstruction_loss
        # self.hyper_params['recloss'] = self.train_config.ae_reconstruction_loss
        self.hyper_params['specmindB'] = self.model_config.spectrogram_min_dB
        self.hyper_params['mel_nbins'] = self.model_config.mel_bins
        self.hyper_params['mel_fmin'] = self.model_config.mel_f_limits[0]
        self.hyper_params['mel_fmax'] = self.model_config.mel_f_limits[1]
        # Easily improved tensorboards hparams logging: convert bools to strings
        for k, v in self.hyper_params.items():
            if isinstance(v, bool):
                self.hyper_params[k] = str(v)
                self.hparams_domain_discrete[k] = ['True', 'False']

    def init_hparams_and_metrics(self, metrics):
        """ Hparams and Metric initialization. Will pass if training resumes from saved checkpoint.
        Hparams will be definitely set but metrics can be updated during training.

        :param metrics: Dict of BufferedMetric
        """
        if not self.resume_from_checkpoint:  # tensorboard init at epoch 0 only
            # Some processing on hparams can be done here... none at the moment
            self.update_metrics(metrics)

    def update_metrics(self, metrics):
        """ Updates Tensorboard metrics

        :param metrics: Dict of values and/or BufferedMetric instances
        :return: None
        """
        metrics_dict = dict()
        for k, metric in metrics.items():
            if isinstance(metrics[k], BufferedMetric):
                try:
                    metrics_dict[k] = metric.mean
                except ValueError:
                    metrics_dict[k] = 0  # TODO appropriate default metric value?
            else:
                metrics_dict[k] = metric
        self.add_hparams(self.hyper_params, metrics_dict, hparam_domain_discrete=self.hparams_domain_discrete)

    def add_latent_histograms(self, latent_metric: LatentMetric, dataset_type: str, global_step: int, bins='fd'):
        """
        Adds histograms related to z0 and zK samples to Tensorboard.

        :param dataset_type: 'Train', 'Valid', ...
        """
        z0 = latent_metric.get_z('z0').flatten()
        zK = latent_metric.get_z('zK').flatten()
        self.add_histogram("z0/{}".format(dataset_type), z0, global_step=global_step, bins=bins)
        self.add_histogram("zK/{}".format(dataset_type), zK, global_step=global_step, bins=bins)
        # also add no-outlier histograms (the other ones are actually unreadable...)
        self.add_histogram("z0/{}_no_outlier".format(dataset_type), utils.stat.remove_outliers(z0),
                           global_step=global_step, bins=bins)
        self.add_histogram("zK/{}_no_outlier".format(dataset_type), utils.stat.remove_outliers(zK),
                           global_step=global_step, bins=bins)

