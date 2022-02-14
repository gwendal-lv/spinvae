"""
Allows easy modification of all configuration parameters required to define,
train or evaluate a model.
This script is not intended to be run, it only describes parameters.
However, some dynamic hyper-parameters are properly set when this module is imported.

This configuration is used when running train.py as main.
When running train_queue.py, configuration changes are relative to this config.py file.

When a run starts, this file is stored as a config.json file. To ensure easy restoration of
parameters, please only use simple types such as string, ints, floats, tuples (no lists) and dicts.
"""


import datetime
from utils.config import _Config  # Empty class - to ease JSON serialization of this file


# ===================================================================================================================
# ================================================= Model configuration =============================================
# ===================================================================================================================
model = _Config()

# ----------------------------------------------- Data ---------------------------------------------------
model.data_root_path = "/media/gwendal/Data/Datasets"  # TODO anonymize path
model.logs_root_dir = "saved"  # Path from this directory
model.name = "FlowInterp"
model.run_name = 'dummy_test'  # run: different hyperparams, optimizer, etc... for a given model
model.allow_erase_run = True  # If True, a previous run with identical name will be erased before training
model.pretrained_VAE_checkpoint = "/home/gwendal/Jupyter/nn-synth-interp/saved/" \
                                  "VAE_MMD_5020/presets_x4__enc_big_dec3resblk__batch64/checkpoints/00399.tar"

# ---------------------------------------- General Architecture --------------------------------------------
# See model/encoder.py to view available architectures. Decoder architecture will be as symmetric as possible.
# 'speccnn8l1' used for the DAFx paper (based on 4x4 kernels, square-shaped deep feature maps)
# 'sprescnn': Spectral Res-CNN (based on 1x1->3x3->1x1 res conv blocks)
# Arch args:
#    '_adain' some BN layers are replaced by AdaIN (fed with a style vector w, dim_w < dim_z)
#    '_att' self-attention in deep conv layers  TODO encoder and decoder
#    '_big' (small improvements but +50% GPU RAM usage),   '_bigger'
#    '_res' residual connections (blocks of 2 conv layer)
#    '_time+' increases time resolution in the deepest layers
model.encoder_architecture = 'speccnn8l1_res_big'
model.attention_gamma = 1.0  # Amount of self-attention added to (some) usual convolutional outputs
# Style network architecture: to get a style vector w from a sampled latent vector z0 (inspired by StyleGAN)
# must be an mlp, but the number of layers and output normalization (_outputbn) can be configured
# e.g. 8l1024: 8 layers, 1024 units per layer
model.style_architecture = 'mlp_2l128_outputbn'  # batch norm layers are always added inside the mlp
# Possible values: 'flow_realnvp_6l300', 'mlp_3l1024', ... (configurable numbers of layers and neurons)
# TODO random permutations when building flows
# 3l600 is associated to bad MMD values and "dirac-like" posteriors.
# Maybe try a bigger flow to prevent a to strong constraint on the latent space?
model.params_regression_architecture = 'flow_realnvp_8l500'  # TODO try bigger flow (if does not overfit anymore)
model.params_reg_hardtanh_out = False  # Applies to categorical params (numerical are always hardtanh-activated)
model.params_reg_softmax = False  # Apply softmax at the end of the reg model itself?
# If True, loss compares v_out and v_in. If False, we will flow-invert v_in to get loss in the q_Z0 domain.
# This option has implications on the regression model itself (the flow will be used in direct or inverse order)
model.forward_controls_loss = True  # Must be true for non-invertible MLP regression (False is now deprecated)

# --------------------------------------------- Latent space -----------------------------------------------
# If True, encoder output is reduced by 2 for 1 MIDI pitch and 1 velocity to be concatenated to the latent vector
model.concat_midi_to_z = None  # See update_dynamic_config_params()
# Latent space dimension  *************** When using a Flow regressor, this dim is automatically set ******************
model.dim_z = 5020  # Including possibly concatenated midi pitch and velocity
# Latent flow architecture, e.g. 'realnvp_4l200' (4 flows, 200 hidden features per flow)
#    - base architectures can be realnvp, maf, ...
#    - set to None to disable latent space flow transforms: will build a BasicVAE or MMD-VAE
#    - options: _BNinternal (batch norm between hidden MLPs, to compute transform coefficients),
#               _BNbetween (between flow layers), _BNoutput (BN on the last two layers, or not)
model.latent_flow_arch = None
# model.latent_flow_arch = 'realnvp_6l300_BNinternal_BNbetween'

# ------------------------------------------------ Audio -------------------------------------------------
# Spectrogram size cannot easily be modified - all CNN decoders should be re-written
model.note_duration = (3.0, 1.0)
model.sampling_rate = 16000  # 16000 for NSynth dataset compatibility
model.stft_args = (512, 256)  # fft size and hop size
model.mel_bins = -1  # -1 disables Mel-scale spectrogram. Try: 257, 513, ...
# Spectrogram sizes @ 22.05 kHz:
#   (513, 433): audio 5.0s, fft size 1024, fft hop 256
#   (257, 347): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
#   (513, 347): audio 4.0s, fft size 1024 (no mel), fft hop 256
# Sizes @ 16 kHz:
#   (257, 251): audio 4.0s, fft size 512 (or fft 1024 w/ mel_bins 257), fft hop 256
model.spectrogram_size = (257, 251)  # H x W. see data/dataset.py to retrieve this from audio/stft params
model.mel_f_limits = (0, 8000)  # min/max Mel-spectrogram frequencies (librosa default 0:Fs/2)
# All the notes that must be available for each instrument (even if we currently use only a subset of those notes)
model.required_dataset_midi_notes = ((41, 75), (48, 75), (56, 75), (63, 75), (56, 25), (56, 127))
# Tuple of (pitch, velocity) tuples. Using only 1 midi note is fine.
# model.midi_notes = ((56, 75), )  # Reference note: G#3 , intensity 75/127
model.midi_notes = model.required_dataset_midi_notes
model.stack_spectrograms = True  # If True, dataset will feed multi-channel spectrograms to the encoder
model.stack_specs_features_mix_level = -2  # -1 corresponds to the deepest 1x1 conv, -2 to the layer before, ...
# If True, each preset is presented several times per epoch (nb of train epochs must be reduced) such that the
# dataset size is artificially increased (6x bigger with 6 MIDI notes) -> warmup and patience epochs must be scaled
model.increased_dataset_size = None  # See update_dynamic_config_params()
model.spectrogram_min_dB = -120.0
model.input_tensor_size = None  # see update_dynamic_config_params()

# -------------------------------------- Synth (not used during pre-training) --------------------------------------
model.synth = 'dexed'
# Dexed-specific auto rename: '*' in 'al*_op*_lab*' will be replaced by the actual algorithms, operators and labels
model.synth_args_str = 'al*_op*_lab*'  # Auto-generated string (see end of script)
model.synth_params_count = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
model.learnable_params_tensor_length = -1  # Will be set automatically - see data.build.get_full_and_split_datasets
# Modeling of synth controls probability distributions
# Possible values: None, 'vst_cat', 'all<=xx' where xx is numerical params threshold cardinal, or 'all'
model.synth_vst_params_learned_as_categorical = 'all'
model.continuous_params_max_resolution = 50  # resolution of continuous synth params will be reduced to this value
# flags/values to describe the dataset to be used
model.dataset_labels = None  # tuple of labels (e.g. ('harmonic', 'percussive')), or None to use all available labels
# Dexed: Preset Algorithms, and activated Operators (Lists of ints, None to use all)
# Limited algorithms (non-symmetrical only): [1, 2, 7, 8, 9, 14, 28, 3, 4, 11, 16, 18]
# Other synth: ...?
model.dataset_synth_args = (None, [1, 2, 3, 4, 5, 6])
# Directory for saving metrics, samples, models, etc... see README.md


# ===================================================================================================================
# ======================================= Training procedure configuration ==========================================
# ===================================================================================================================
train = _Config()
train.pretrain_ae_only = False  # Should we pre-train the auto-encoder model only?
train.start_datetime = datetime.datetime.now().isoformat()
train.minibatch_size = 64  # 128: faster train but lower higher MMD (more posterior collapse)
train.main_cuda_device_idx = 0  # CUDA device for nonparallel operations (losses, ...). -1 indicates CPU (untested)
train.test_holdout_proportion = 0.1  # This can be reduced without mixing the train and test subsets
train.k_folds = 9  # 10% for validation set, 80% for training
train.current_k_fold = 0
train.start_epoch = 0  # 0 means a restart (previous data erased). If > 0: will load start_epoch-1 checkpoint
# Total number of epochs (including previous training epochs).  275 for StepLR regression model training
train.n_epochs = 400 if train.pretrain_ae_only else 275  # See update_dynamic_config_params().
# The max ratio between the number of items from each synth/instrument used for each training epoch (e.g. Dexed has
# more than 30x more instruments than NSynth). All available data will always be used for validation.
train.pretrain_synths_max_imbalance_ratio = 10.0  # Set to -1 to disable the weighted sampler.
train.attention_gamma_warmup_period = 50

# ------------------------------------------------ Losses -------------------------------------------------
# Reconstruction loss: 'MSE' corresponds to free-mean, fixed-variance per-pixel Gaussian probability distributions.
# TODO 'WeightedMSE' allows to give a higher loss to some parts of spectrograms (e.g. attach, low freqs, ??)
train.reconstruction_loss = 'MSE'
# Latent regularization loss: 'Dkl' or 'MMD' for Basic VAE, 'logprob' or 'MMD' loss with flow-VAE
# 'MMD_determ_enc' also available: use a deterministic encoder
train.latent_loss = 'MMD'
train.mmd_compensation_factor = 5.0  # Factor applied to MMD backprop losses only
train.mmd_num_estimates = 1  # Number of MMD estimates per batch (maybe increase if small batch size)
# Losses normalization allow to get losses in the same order of magnitude, but does not optimize the true ELBO.
# When un-normalized, the reconstruction loss (log-probability of a multivariate gaussian) is orders of magnitude
# bigger than other losses. Train does not work with normalize=False at the moment - use train.beta to compensate
train.normalize_losses = True  # Normalize all losses over the vector-dimension (e.g. spectrogram pixels count, D, ...)
# (beta<1, normalize=True) corresponds to (beta>>1, normalize=False) in the beta-VAE formulation (ICLR 2017)
train.beta = 0.2  # latent loss factor (base value: 0.2) - use much lower value (e-2) to get closer the ELBO
train.beta_start_value = train.beta / 2.0  # Should not be zero (risk of a very unstable training)
# Epochs of warmup increase from start_value to beta
train.beta_warmup_epochs = 25  # See update_dynamic_config_params(). Used during pre-train only
# - - - Synth parameters losses - - -
# - General options
train.params_model_additional_regularization = None  # 'inverse_log_prob' available for Flow-based models
train.params_loss_compensation_factor = 1.0  # because MSE loss of the pre-trained VAE if much lower (approx. 1e-2)
train.params_loss_exclude_useless = True  # if True, sets to the 0.0 the loss related to 0-volume oscillators
train.params_loss_with_permutations = False  # Applies to the backprop loss only; monitoring losses always use True
# - Loss for a dense dequantized output loss (set to 'None' to activate other losses)
train.params_dense_dequantized_loss = 'None'  # Preempts CE losses
# - Cross-Entropy loss (deactivated when using dequantized outputs)
train.params_cat_CE_label_smoothing = 0.0  # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
train.params_target_noise = 0.00
train.params_cat_CE_use_weights = False
train.params_cat_bceloss = False  # If True, disables the CE loss to compute BCE loss instead (deprecated)
train.params_cat_softmax_temperature = 1.0  # Temperature if softmax if applied in the loss only (!=0 is deprecated)

# ------------------------------------------- Optimizer + scheduler -------------------------------------------
# Different optimizers for the pre-trained AE and the regression networks ('ae_' or 'reg_' prefixes or dict keys)
train.optimizer = 'Adam'
# Maximal learning rate (reached after warmup, then reduced on plateaus)
# LR decreased if non-normalized losses (which are expected to be 90,000 times bigger with a 257x347 spectrogram)
# e-9 LR with e+4 (non-normalized) loss does not allow any train (vanishing grad?)
train.initial_learning_rate = {'ae': 1e-4, 'reg': 1e-4}
train.initial_ae_lr_factor_after_pretrain = 1e-1  # AE LR reduced when used with regression model after pre-train
# Learning rate warmup (see https://arxiv.org/abs/1706.02677). Same warmup period for all schedulers.
# The warmup will be must faster during pre-train  (See update_dynamic_config_params())
train.lr_warmup_epochs = 20
train.lr_warmup_start_factor = 0.05  # Reduced for large realnvp flows
train.adam_betas = (0.9, 0.999)  # default (0.9, 0.999)
train.scheduler_name = 'StepLR'  # use ReduceLROnPlateau during pre-train (stable CNN), StepLR for reg model
train.enable_ae_scheduler_after_pretrain = False
train.scheduler_lr_factor = {'ae': 0.4, 'reg': 0.2}
# - - - StepLR scheduler options - - -
train.scheduler_period = 50  # Will be increased during pre-train
# - - - ReduceLROnPlateau scheduler options - - -
# Possible values: 'VAELoss' (total), 'ReconsLoss', 'Controls/BackpropLoss', ... All required losses will be summed
train.scheduler_losses = {'ae': ('ReconsLoss/Backprop', ), 'reg': ('Controls/BackpropLoss', )}
# Set a longer patience with smaller datasets and quite unstable trains
# See update_dynamic_config_params(). 16k samples dataset:  set to 10
train.scheduler_patience = {'ae': 25, 'reg': 15}
train.scheduler_cooldown = {'ae': 25, 'reg': 15}
train.scheduler_threshold = 1e-4
# Training considered "dead" when dynamic LR reaches this value (or the initial LR multiplied by the following ratios)
# Early stop is currently used for the regression loss only, for the 'ReduceLROnPlateau' scheduler only.
train.early_stop_lr_ratio = {'ae': 1e-10, 'reg': 1e-4}  # early stop not implemented for the ae model
train.early_stop_lr_threshold = None  # See update_dynamic_config_params()

# ----------------------------------------------- Regularization --------------------------------------------------
# WD definitely helps for regularization but significantly impairs results. 1e-4 seems to be a good compromise
# for both Basic and MMD VAEs (without regression net). 3e-6 allows for the lowest reconstruction error.
train.weight_decay = 1e-5
train.fc_dropout = 0.0  # 0.3 without MMD, to try to help prevent VAE posterior collapse
train.reg_fc_dropout = 0.4
train.latent_input_dropout = 0.0  # Should always remain zero... intended for tests (not tensorboard-logged)
# When using a latent flow z0-->zK, z0 is not regularized. To keep values around 0.0, batch-norm or a 0.1Dkl can be used
# (warning: latent input batch-norm is a very strong constraint for the network)
# 'BN' (on encoder output), 'Dkl' (on q_Z0 gaussian flow input) or 'None' (always use a str arg)
train.latent_flow_input_regularization = 'None'
train.latent_flow_input_regul_weight = 0.1  # Used for 'Dkl' only

# -------------------------------------------- Logs, figures, ... ---------------------------------------------
train.save_period = 500  # Period for checkpoint saves (large disk size)
train.plot_period = 20   # Period (in epochs) for plotting graphs into Tensorboard (quite CPU and SSD expensive)
train.verbosity = 1  # 0: no console output --> 3: fully-detailed per-batch console output
train.init_security_pause = 0.0  # Short pause before erasing an existing run
# Number of logged audio and spectrograms for a given epoch
train.logged_samples_count = 4  # See update_dynamic_config_params()

# ------------------------------------------ Performance and Profiling ----------------------------------------------
train.dataloader_pin_memory = False
train.dataloader_persistent_workers = True
train.profiler_enabled = False
train.profiler_epoch_to_record = 0  # The profiler will record a few minibatches of this given epoch
train.profiler_kwargs = {'record_shapes': True, 'with_stack': True}
train.profiler_schedule_kwargs = {'skip_first': 5, 'wait': 1, 'warmup': 1, 'active': 3, 'repeat': 2}





def update_dynamic_config_params():  # Required before any actual train
    """
    Updates some global attributes of this config.py module.
    This function should be called before using any train attribute
    """

    if train.pretrain_ae_only:
        model.params_regression_architecture = 'None'
        train.lr_warmup_epochs = train.lr_warmup_epochs // 2
        train.lr_warmup_start_factor *= 2
        train.scheduler_period += train.scheduler_period // 2
    else:
        train.initial_learning_rate['ae'] *= train.initial_ae_lr_factor_after_pretrain
        train.beta_warmup_epochs = 0

    # stack_spectrograms must be False for 1-note datasets - security check
    model.stack_spectrograms = model.stack_spectrograms and (len(model.midi_notes) > 1)
    # Artificially increased data size?
    model.increased_dataset_size = (len(model.midi_notes) > 1) and not model.stack_spectrograms
    model.concat_midi_to_z = (len(model.midi_notes) > 1) and not model.stack_spectrograms
    # Mini-batch size can be smaller for the last mini-batches and/or during evaluation
    model.input_tensor_size = (train.minibatch_size, 1 if not model.stack_spectrograms else len(model.midi_notes),
                               model.spectrogram_size[0], model.spectrogram_size[1])

    # Dynamic train hyper-params
    train.early_stop_lr_threshold = {k: train.initial_learning_rate[k] * ratio
                                     for k, ratio in train.early_stop_lr_ratio.items()}
    train.logged_samples_count = max(train.logged_samples_count, len(model.midi_notes))
    # Train hyper-params (epochs counts) that should be increased when using a subset of the dataset
    if model.dataset_synth_args[0] is not None:  # Limited Dexed algorithms?  TODO handle non-dexed synth
        train.n_epochs = 700
        train.lr_warmup_epochs = 10
        train.scheduler_patience = 10
        train.scheduler_cooldown = 10
        train.beta_warmup_epochs = 40
    # Train hyper-params (epochs counts) that should be reduced with artificially increased datasets
    # Augmented  datasets introduce 6x more backprops <=> 6x more epochs. Patience and cooldown must however remain >= 2
    if model.increased_dataset_size:  # Stacked spectrogram do not increase the dataset size (number of items)
        # FIXME handle the dicts
        N = len(model.midi_notes) - 1  # reduce a bit less that dataset's size increase
        train.n_epochs = 1 + train.n_epochs // N
        train.lr_warmup_epochs = 1 + train.lr_warmup_epochs // N
        train.scheduler_patience = 1 + train.scheduler_patience // N
        train.scheduler_cooldown = 1 + train.scheduler_cooldown // N
        train.beta_warmup_epochs = 1 + train.beta_warmup_epochs // N

    # Automatic model.synth string update - to summarize this info into 1 Tensorboard string hparam
    if model.synth == "dexed":
        if model.dataset_synth_args[0] is not None:  # Algorithms
            model.synth_args_str = model.synth_args_str.replace("al*", "al" +
                                                                '.'.join(
                                                                    [str(alg) for alg in model.dataset_synth_args[0]]))
        if model.dataset_synth_args[1] is not None:  # Operators
            model.synth_args_str = model.synth_args_str.replace("_op*", "_op" +
                                                                ''.join(
                                                                    [str(op) for op in model.dataset_synth_args[1]]))
        if model.dataset_labels is not None:  # Labels
            model.synth_args_str = model.synth_args_str.replace("_lab*", '_' +
                                                                '_'.join(
                                                                    [label[0:4] for label in model.dataset_labels]))
    else:
        raise NotImplementedError("Unknown synth prefix for model.synth '{}'".format(model.synth))

