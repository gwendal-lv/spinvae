"""
Allows easy modification of all configuration parameters required to define,
train or evaluate a model.
This script is not intended to be run, it only describes parameters (see classes constructors).
After building the config instances, the update_dynamic_config_params(...) method
must be called to update some "dynamic" hyper-parameters which depend on some others.

This configuration is used when running train.py as main.
When running train_queue.py, configuration changes are relative to this config.py file.

When a run starts, this file is stored as a config.json file. To ensure easy restoration of
parameters, please only use simple types such as string, ints, floats, tuples (no lists) and dicts.
"""


import datetime
import pathlib

# The config_confidential.py file must be created by the user in the ./utils folder. It must contain
# a few global attributes which are described later in this file (see ModelConfig ctor)
from utils import config_confidential


# ===================================================================================================================
# ================================================= Model configuration =============================================
# ===================================================================================================================
class ModelConfig:
    def __init__(self):
        # ----------------------------------------------- Data ---------------------------------------------------
        self.data_root_path = config_confidential.data_root_path
        self.logs_root_dir = config_confidential.logs_root_dir
        self.name = "dev"  # experiment base name
        # experiment run: different hyperparams, optimizer, etc... for a given exp
        self.run_name = 'combined_vae__fftoken_NOnorm'
        self.pretrained_VAE_checkpoint = \
            self.logs_root_dir + "/hvae/8x1_freebits0.250__6notes_dimz256/checkpoint.tar"
            #self.logs_root_dir + "/hvae/8x1_freebits0.125__3notes_dimz256/checkpoint.tar"
        # self.pretrained_VAE_checkpoint = None  # Uncomment this to train a full model from scratch
        self.allow_erase_run = True  # If True, a previous run with identical name will be erased before training
        # Comet.ml logger (replaces Tensorboard)
        self.comet_api_key = config_confidential.comet_api_key
        self.comet_project_name = config_confidential.comet_project_name
        self.comet_workspace = config_confidential.comet_workspace
        self.comet_experiment_key = 'xxxxxxxx'  # Will be set by cometwriter.py after experiment has been created
        self.comet_tags = []

        # ---------------------------------------- General Architecture --------------------------------------------
        # See model/encoder.py to view available architectures. Decoder architecture will be as symmetric as possible.
        # 'speccnn8l' used for the DAFx paper (based on 4x4 kernels, square-shaped deep feature maps)
        # 'sprescnn': Spectral Res-CNN (based on 1x1->3x3->1x1 res conv blocks)
        # 'specladder': ladder CNNs (cells, outputs from different levels) for spectrogram reconstruction
        #       also contains num of blocks and num of conv layers per block (e.g. 8x1)
        # Arch args:
        #    '_adain' some BN layers are replaced by AdaIN (fed with a style vector w, dim_w < dim_z)
        #    '_att' self-attention in deep conv layers
        #    '_big' (small improvements but +50% GPU RAM usage),   '_bigger'
        #    '_res' residual connections after each hidden strided conv layer (up/down sampling layers)
        #    '_depsep5x5' uses 5x5 depth-separable convolutional layers in each res block (requires at least 8x2)
        #    '_ln' uses LayerNorm instead of BatchNorm, '_wn' uses Weight Normalization attached to conv weights
        #    '_swish' uses Swish activations (SiLU) instead of LeakyReLU (negligible overhead)
        self.vae_main_conv_architecture = 'specladder8x1_res_swish'  # should be named "audio" architecture...
        # Network plugged after sequential conv blocks (encoder) or before sequential conv blocks (decoder)
        # E.g.: 'conv_1l_1x1' means regular convolution, 1 layer, 1x1 conv kernels
        #       'lstm_2l_3x3' mean ConvLSTM, 2 layers, 3x3 conv kernels
        # other args: '_gated' (for regular convolutions) seems very effective to improve the overall ELBO
        #             '_att' adds self-attention conv residuals at the beginning of shallow latent cells
        #             '_posenc' adds input positional information to LSTMs and to self-attention conv layers
        self.vae_latent_extract_architecture = 'conv_1l_k1x1_gated'
        # Number of latent levels increases the size of the shallowest latent feature maps, and increases the
        # minimal dim_z requirement
        #   2 latent levels allows dim_z close to 100
        #   3 latent levels allows dim_z close to 350
        #   4 latent levels allows dim_z close to 1500
        self.vae_latent_levels = 1  # Currently useless, must remain 1
        # Sets the family of decoder output probability distribution p_theta(x|z), e.g. :
        #    - 'gaussian_unitvariance' corresponds to the usual MSE reconstruction loss (up to a constant and factor)
        self.audio_decoder_distribution = 'gaussian_unitvariance'
        self.attention_gamma = 1.0  # Amount of self-attention added to (some) usual convolutional outputs
        # Preset encoder/decoder architecture
        # TODO description (base + options)
        #   '_ff': feed-forward, non-AR decoding - applicable to sequential models: RNN, Transformer (pos enc only)
        #   '_memmlp': doubles the number of Transformer decoder memory tokens using a "Res-MLP" on the latent vector
        #              -> seems to improves perfs a bit (lower latent loss, quite similar auto synth prog losses)
        self.vae_preset_architecture = 'tfm_6l_ff_memmlp_fftoken'  # tfm_6l_memmlp_ff
        # "before_latent_cell" (encoded preset will be the same size as encoded audio, both will be added)
        # or "after_latent_cell"" (encoded preset size will be 2*dim_z, and will be added to z_mu and z_sigma)
        self.vae_preset_encode_add = "after_latent_cell"
        # Size of the hidden representation of 1 synth parameter
        self.preset_hidden_size = 256
        # Distribution for modeling (discrete-)numerical synth param values.
        # (categorical variables always use a softmaxed categorical distribution)
        self.preset_decoder_numerical_distribution = 'logistic_mixt3'
        # Describes how (if) the presets should be auto-encoded:
        #    - "no_encoding": presets are inferred from audio but are not encoded (not provided at
        #               encoder input). Corresponds to an ASP (Automatic Synthesizer Programming) situation.
        #    - "combined_vae": preset is encoded with audio, their hidden representations are then summed or mixed
        #           together
        #    - TODO "asp+vae": hybrid method/training: TODO DOC
        #    - "independent_vae": the preset VAE and audio VAE are trained as independent models, but a loss
        #           (e.g. contrastive, Dkl, ... TODO TBD) is computed using the two latent representations
        #    - "no_audio": the preset alone is auto-encoded, audio is discarded
        self.preset_ae_method = "combined_vae"

        # --------------------------------------------- Latent space -----------------------------------------------
        # If True, encoder output is reduced by 2 for 1 MIDI pitch and 1 velocity to be concat to the latent vector
        self.concat_midi_to_z = None  # See update_dynamic_config_params()
        # Latent space dimension  ********* this dim is automatically set when using a Hierarchical VAE *************
        self.dim_z = -1
        self.approx_requested_dim_z = 256  # Hierarchical VAE will try to get close to this, will often be higher
        # Latent flow architecture, e.g. 'realnvp_4l200' (4 flows, 200 hidden features per flow)
        #    - base architectures can be realnvp, maf, ...
        #    - set to None to disable latent space flow transforms: will build a BasicVAE or MMD-VAE
        #    - options: _BNinternal (batch norm between hidden MLPs, to compute transform coefficients),
        #               _BNbetween (between flow layers), _BNoutput (BN on the last two layers, or not)
        self.latent_flow_arch = None
        # self.latent_flow_arch = 'realnvp_6l300_BNinternal_BNbetween'

        # ------------------------------------------------ Audio -------------------------------------------------
        # Spectrogram size cannot easily be modified - all CNN decoders should be re-written
        self.note_duration = (3.0, 1.0)
        self.sampling_rate = 16000  # 16000 for NSynth dataset compatibility
        self.stft_args = (512, 256)  # fft size and hop size
        self.mel_bins = -1  # -1 disables Mel-scale spectrogram. Try: 257, 513, ...
        # Spectrogram sizes @ 22.05 kHz:
        #   (513, 433): audio 5.0s, fft size 1024, fft hop 256
        #   (257, 347): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
        #   (513, 347): audio 4.0s, fft size 1024 (no mel), fft hop 256
        # Sizes @ 16 kHz:
        #   (257, 251): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
        self.spectrogram_size = (257, 251)  # H x W. see data/dataset.py to retrieve this from audio/stft params
        self.mel_f_limits = (0, 8000)  # min/max Mel-spectrogram frequencies (librosa default 0:Fs/2)
        # All notes that must be available for each instrument (even if we currently use only a subset of those notes)
        self.required_dataset_midi_notes = ((41, 75), (48, 75), (56, 75), (63, 75), (56, 25), (56, 127))
        # Tuple of (pitch, velocity) tuples. Using only 1 midi note is fine.
        # self.midi_notes = ((56, 75), )  # Reference note: G#3 , intensity 75/127
        self.midi_notes = ((41, 75), (48, 75), (56, 25), (56, 75), (56, 127), (63, 75))  # 6 notes
        # self.midi_notes = ((41, 75), (56, 25), (56, 75), (56, 127), (63, 75))  # 5 notes
        # self.midi_notes = ((41, 75), (56, 75), (56, 127))  # 3 notes (faster training)
        self.main_midi_note_index = len(self.midi_notes) // 2  # 56, 75
        self.stack_spectrograms = True  # If True, dataset will feed multi-channel spectrograms to the encoder
        # If True, each preset is presented several times per epoch (nb of train epochs must be reduced) such that the
        # dataset size is increased (6x bigger with 6 MIDI notes) -> warmup and patience epochs must be scaled
        self.increased_dataset_size = None  # See update_dynamic_config_params()
        self.spectrogram_min_dB = -120.0
        self.input_audio_tensor_size = None  # see update_dynamic_config_params()

        # ---------------------------------- Synth (not used during pre-training) ----------------------------------
        self.synth = 'dexed'
        # Dexed-specific auto rename: '*' in 'al*_op*_lab*' will be replaced by the actual algos, operators and labels
        self.synth_args_str = 'al*_op*_lab*'  # Auto-generated string (see end of script)
        self.synth_params_count = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
        # FIXME Modeling of synth controls probability distributions - RE-IMPLEMENT, SHOULD BE A MODEL ARGUMENT
        # flags/values to describe the dataset to be used
        self.dataset_labels = None  # tuple of labels (e.g. ('harmonic', 'percussive')), or None to use all labels
        # Dexed: Preset Algorithms, and activated Operators (Lists of ints, None to use all)
        # Limited algorithms (non-symmetrical only): [1, 2, 7, 8, 9, 14, 28, 3, 4, 11, 16, 18]
        # Other synth: ...?
        self.dataset_synth_args = (None, [1, 2, 3, 4, 5, 6])
        # Directory for saving metrics, samples, models, etc... see README.md


# ===================================================================================================================
# ======================================= Training procedure configuration ==========================================
# ===================================================================================================================
class TrainConfig:
    def __init__(self):
        self.pretrain_audio_only = False  # Should we pre-train the audio+latent parts of the auto-encoder model only?
        self.start_datetime = datetime.datetime.now().isoformat()
        # 256 is okay for smaller conv structures - reduce to 64 to fit '_big' models into 24GB GPU RAM
        self.minibatch_size = 64  # reduce for big models - also smaller N seems to improve VAE pretraining perfs...
        self.main_cuda_device_idx = 0  # CUDA device for nonparallel operations (losses, ...)
        self.test_holdout_proportion = 0.1  # This can be reduced without mixing the train and test subsets
        self.k_folds = 9  # 10% for validation set, 80% for training
        self.current_k_fold = 0  # k-folds are not used anymore, but we'll keep the training/validation/test splits
        self.start_epoch = 0  # 0 means a restart (previous data erased). If > 0: will load the last saved checkpoint
        # Total number of epochs (including previous training epochs).  275 for StepLR regression model training
        self.n_epochs = 170  # See update_dynamic_config_params().
        # The max ratio between the number of items from each synth/instrument used for each training epoch (e.g. Dexed
        # has more than 30x more instruments than NSynth). All available data will always be used for validation.
        self.pretrain_synths_max_imbalance_ratio = 10.0  # Set to -1 to disable the weighted sampler.
        self.attention_gamma_warmup_period = 50

        # ------------------------------------------------ Losses -------------------------------------------------
        # Reconstruction loss: 'MSE' corresponds to free-mean, fixed-variance per-pixel Gaussian prob distributions.
        # TODO 'WeightedMSE' allows to give a higher loss to some parts of spectrograms (e.g. attack, low freqs, ??)
        self.reconstruction_loss = 'MSE'
        # Latent regularization loss: 'Dkl' or 'MMD' for Basic VAE, 'logprob' or 'MMD' loss with flow-VAE
        # 'MMD_determ_enc' also available: use a deterministic encoder
        self.latent_loss = 'Dkl'
        self.mmd_compensation_factor = 5.0  # Factor applied to MMD backprop losses only
        self.mmd_num_estimates = 1  # Number of MMD estimates per batch (maybe increase if small batch size)
        # Losses normalization allow to get losses in the same order of magnitude, but does not optimize the true ELBO.
        # When un-normalized, the reconstruction loss (log-probability of a multivariate gaussian) is orders of
        # magnitude bigger than other losses. Must remain True to ease convergence (too big recons loss)
        self.normalize_losses = True  # Normalize reconstruction and regression losses over their dimension
        # To compare different latent sizes, Dkl or MMD losses are not normalized such that each latent
        # coordinate always 'has' the same amount of regularization
        self.normalize_latent_loss = False
        # Here, beta = beta_vae / Dx in the beta-VAE formulation (ICLR 2017)
        # where Dx is the input dimensionality (257 * 251 = 64 507 for 1 spectrogram)
        # E.g. here: beta = 1 corresponds to beta_VAE = 6.5 e+4
        #            ELBO loss is obtained by using beta = 1.55 e-5 (for 1 spectrogram)
        self.beta = 1.6e-5  # FIXME With 6 specs, this corresponds to beta=6
        # Should not be zero with Normalizing Flows or with a multi-layer structure for extracting latent values
        # (risk of a very unstable training)
        self.beta_start_value = self.beta * 1e-3
        # Epochs of warmup increase from start_value to beta TODO increase to reduce posterior collapse
        self.beta_warmup_epochs = 50  # Used during both pre-train and fine-tuning
        # VAE Kullback-Leibler divergence weighting during warmup, to try to prevent posterior collapse of some
        # latent levels (see NVAE, NeurIPS 2020). Leads to higher latent losses during warmup.
        self.dkl_auto_gamma = False  # If True, latent groups with small minibatch-KLDs will be assigned a smaller loss
        # Free-bits from the IAF paper (NeurIPS 16) https://arxiv.org/abs/1606.04934
        # Our VAE uses 2D feature maps as latent variables, and entire channels seem to collapse
        # (a single pixel collapsing has not been observed). The "free bits" min Dkl constraint is then applied
        # to each latent channel, no to hierarchical latent groups
        # TODO increase? collapse during fine-tuning   TODO upaate doc
        self.latent_free_bits = 0.250  # this is a *single-pixel* min KLD value

        # - - - Synth parameters losses - - -
        # - General options
        self.params_model_additional_regularization = None  # 'inverse_log_prob' available for Flow-based models
        # applied to the preset loss FIXME because MSE loss of the VAE is much lower (approx. 1e-2)
        self.params_loss_compensation_factor = 0.5
        self.params_loss_exclude_useless = False  # if True, sets to 0.0 the loss related to 0-volume oscillators
        self.params_loss_with_permutations = False  # Backprop loss only; monitoring losses always use True
        # - Cross-Entropy loss (deactivated when using dequantized outputs)
        self.preset_CE_label_smoothing = 0.1  # torch.nn.CrossEntropyLoss: label smoothing since PyTorch 1.10
        self.preset_CE_use_weights = False
        # Probability to use the model's outputs during training (AR decoder)
        self.preset_sched_sampling_max_p = 0.0  # Set to zero for FF decoder
        # self.preset_sched_sampling_start_epoch = 40  # TODO IMPLEMENT Required for the embeddings to train properly?
        self.preset_sched_sampling_warmup_epochs = 100

        # ------------------------------------------- Optimizer + scheduler -------------------------------------------
        # Different optimizer parameters can be used for the pre-trained AE and the regression networks
        # (see below: 'ae' or 'reg' prefixes or dict keys)
        self.optimizer = 'Adam'
        self.adam_betas = (0.9, 0.999)  # default (0.9, 0.999)
        # Maximal learning rate (reached after warmup, then reduced on plateaus)
        # LR decreased if non-normalized losses (which are expected to be 9e4 times bigger with a 257x347 spectrogram)
        # e-9 LR with e+4 (non-normalized) loss does not allow any train (vanishing grad?)
        self.initial_learning_rate = {'audio': 2e-4, 'latent': 2e-4, 'preset': 2e-4}
        # FIXME RESET TO 1e-1?
        self.initial_audio_latent_lr_factor_after_pretrain = 1.0  # audio-related LR reduced after pre-train
        # Learning rate warmup (see https://arxiv.org/abs/1706.02677). Same warmup period for all schedulers.
        self.lr_warmup_epochs = 20  # Will be decreased /2 during pre-training (stable CNN structure)
        self.lr_warmup_start_factor = 0.05  # Will be increased 2x during pre-training
        self.scheduler_name = 'StepLR'  # can use ReduceLROnPlateau during pre-train (stable CNN), StepLR for reg model
        self.scheduler_lr_factor = 0.4
        # - - - StepLR scheduler options - - -
        self.scheduler_period = 75  # resnets train quite fast
        # Set a longer patience with smaller datasets and quite unstable trains
        # See update_dynamic_config_params(). 16k samples dataset:  set to 10
        self.scheduler_patience = 20
        self.scheduler_cooldown = 20
        self.scheduler_threshold = 1e-4
        # Training considered "dead" when dynamic LR reaches this ratio of a the initial LR
        # Early stop is currently used for the regression loss only, for the 'ReduceLROnPlateau' scheduler only.
        self.early_stop_lr_ratio = 1e-4
        self.early_stop_lr_threshold = None  # See update_dynamic_config_params()

        # -------------------------------------------- Regularization -----------------------------------------------
        # WD definitely helps for regularization but significantly impairs results. 1e-4 seems to be a good compromise
        # for both Basic and MMD VAEs (without regression net). 3e-6 allows for the lowest reconstruction error.
        self.weight_decay = 1e-5
        self.ae_fc_dropout = 0.0
        # FIXME use dropout
        self.preset_cat_dropout = 0.0  # Applied only to subnets which do not handle value regression tasks
        self.preset_internal_dropout = 0.0

        # -------------------------------------------- Logs, figures, ... ---------------------------------------------
        self.validate_period = 5  # Period between validations (very long w/ autoregressive transformers)
        self.plot_period = 20   # Period (in epochs) for plotting graphs into Tensorboard (quite CPU and SSD expensive)
        self.large_plots_min_period = 100  # Min num of epochs between plots (e.g. embeddings, approx. 80MB .tsv files)
        self.plot_epoch_0 = False
        self.verbosity = 1  # 0: no console output --> 3: fully-detailed per-batch console output
        self.init_security_pause = 0.0  # Short pause before erasing an existing run
        # Number of logged audio and spectrograms for a given epoch
        self.logged_samples_count = 4  # See update_dynamic_config_params()

        # -------------------------------------- Performance and Profiling ------------------------------------------
        self.dataloader_pin_memory = False
        self.dataloader_persistent_workers = True
        self.profiler_enabled = False
        self.profiler_epoch_to_record = 0  # The profiler will record a few minibatches of this given epoch
        self.profiler_kwargs = {'record_shapes': True, 'with_stack': True}
        self.profiler_schedule_kwargs = {'skip_first': 5, 'wait': 1, 'warmup': 1, 'active': 3, 'repeat': 2}





def update_dynamic_config_params(model_config: ModelConfig, train_config: TrainConfig):
    """ This function must be called before using any train attribute """

    # TODO perform config coherence checks in this function

    if train_config.pretrain_audio_only:
        model_config.comet_tags.append('pretrain')
        model_config.params_regression_architecture = 'None'
        train_config.lr_warmup_epochs = train_config.lr_warmup_epochs // 2
        train_config.lr_warmup_start_factor *= 2
    else:
        train_config.initial_learning_rate['audio'] *= train_config.initial_audio_latent_lr_factor_after_pretrain
        train_config.initial_learning_rate['latent'] *= train_config.initial_audio_latent_lr_factor_after_pretrain

    # stack_spectrograms must be False for 1-note datasets - security check
    model_config.stack_spectrograms = model_config.stack_spectrograms and (len(model_config.midi_notes) > 1)
    # Artificially increased data size?
    model_config.increased_dataset_size = (len(model_config.midi_notes) > 1) and not model_config.stack_spectrograms
    model_config.concat_midi_to_z = (len(model_config.midi_notes) > 1) and not model_config.stack_spectrograms
    # Mini-batch size can be smaller for the last mini-batches and/or during evaluation
    model_config.input_audio_tensor_size = \
        (train_config.minibatch_size, 1 if not model_config.stack_spectrograms else len(model_config.midi_notes),
         model_config.spectrogram_size[0], model_config.spectrogram_size[1])

    # Dynamic train hyper-params
    train_config.early_stop_lr_threshold = {k: lr * train_config.early_stop_lr_ratio
                                            for k, lr in train_config.initial_learning_rate.items()}
    train_config.logged_samples_count = max(train_config.logged_samples_count, len(model_config.midi_notes))
    # Train hyper-params (epochs counts) that should be increased when using a subset of the dataset
    if model_config.dataset_synth_args[0] is not None:  # Limited Dexed algorithms?  TODO handle non-dexed synth
        train_config.n_epochs = 700
        train_config.lr_warmup_epochs = 10
        train_config.scheduler_patience = 10
        train_config.scheduler_cooldown = 10
        train_config.beta_warmup_epochs = 40
    # Train hyper-params (epochs counts) that should be reduced with artificially increased datasets
    # Augmented  datasets introduce 6x more backprops <=> 6x more epochs. Patience and cooldown must however remain >= 2
    if model_config.increased_dataset_size:  # Stacked spectrogram do not increase the dataset size (number of items)
        # FIXME handle the dicts
        N = len(model_config.midi_notes) - 1  # reduce a bit less that dataset's size increase
        train_config.n_epochs = 1 + train_config.n_epochs // N
        train_config.lr_warmup_epochs = 1 + train_config.lr_warmup_epochs // N
        train_config.scheduler_patience = 1 + train_config.scheduler_patience // N
        train_config.scheduler_cooldown = 1 + train_config.scheduler_cooldown // N
        train_config.beta_warmup_epochs = 1 + train_config.beta_warmup_epochs // N

    # Automatic model.synth string update - to summarize this info into 1 Tensorboard string hparam
    if model_config.synth == "dexed":
        if model_config.dataset_synth_args[0] is not None:  # Algorithms
            model_config.synth_args_str = model_config.synth_args_str.replace(
                "al*", "al" + '.'.join([str(alg) for alg in model_config.dataset_synth_args[0]]))
        if model_config.dataset_synth_args[1] is not None:  # Operators
            model_config.synth_args_str = model_config.synth_args_str.replace(
                "_op*", "_op" + ''.join([str(op) for op in model_config.dataset_synth_args[1]]))
        if model_config.dataset_labels is not None:  # Labels
            model_config.synth_args_str = model_config.synth_args_str.replace(
                "_lab*", '_' + '_'.join([label[0:4] for label in model_config.dataset_labels]))
    else:
        raise NotImplementedError("Unknown synth prefix for model.synth '{}'".format(model_config.synth))

