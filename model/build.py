"""
Utility functions for building a new model (using only config from config.py),
or for building a previously trained model before loading state dicts.

Decomposed into numerous small function for easier module-by-module debugging.
"""
import warnings

from config import ModelConfig, TrainConfig
from model import VAE, encoder, decoder, extendedAE, regression


def build_encoder_and_decoder_models(model_config: ModelConfig, train_config: TrainConfig):
    # Encoder and decoder with the same architecture
    enc_z_length = (model_config.dim_z - 2 if model_config.concat_midi_to_z else model_config.dim_z)

    encoder_model = encoder.SpectrogramEncoder(
        model_config.vae_main_conv_architecture, model_config.vae_latent_extract_architecture,
        enc_z_length, train_config.latent_loss.endswith('_determ_enc'),
        model_config.input_audio_tensor_size, train_config.ae_fc_dropout,
        output_bn=(train_config.latent_flow_input_regularization.lower() == 'bn'),
        output_dropout_p=train_config.latent_input_dropout,
        deep_features_mix_level=model_config.stack_specs_features_mix_level
    )
    decoder_model = decoder.SpectrogramDecoder(
        model_config.vae_main_conv_architecture,
        model_config.dim_z,model_config.input_audio_tensor_size, train_config.ae_fc_dropout,model_config.midi_notes
    )
    return encoder_model, decoder_model


def build_ae_model(model_config: ModelConfig, train_config: TrainConfig):
    """
    Builds an auto-encoder model given a configuration. Built model can be initialized later
    with a previous state_dict.

    :param model_config: model attributes from the config.py module
    :param train_config: train attributes (a few are required, e.g. dropout probability)
    :return: Tuple: encoder, decoder, full AE model
    """
    raise DeprecationWarning("Deprecated: use Hierarchical VAE constructor")
    # Build encoder and decoder first
    encoder_model, decoder_model = build_encoder_and_decoder_models(model_config, train_config)
    # Backward compatibility - recently added config args
    # Then build the full AE model
    if model_config.latent_flow_arch is None or model_config.latent_flow_arch.lower() == 'none':  # Basic VAE
        # Additional input regularization can't be required for basic VAE (ELBO Dkl regularization term only)
        if train_config.latent_flow_input_regularization != 'None':
            raise AssertionError("BasicVAE: flow input regularization must be 'None' (given arg: '{}')"
                                 .format(train_config.latent_flow_input_regularization))
        if train_config.latent_loss.lower() != 'dkl' and train_config.latent_loss[0:3].lower() != 'mmd':
            raise AssertionError("BasicVAE: the latent loss should be 'Dkl' or 'MMD' (given: '{}')"
                                 .format(train_config.latent_loss))
        ae_model = VAE.BasicVAE(encoder_model, model_config.dim_z, decoder_model, model_config.style_architecture,
                                concat_midi_to_z0=model_config.concat_midi_to_z,  # FIXME deprecated
                                train_config=train_config)
    else:  # Flow VAE
        ae_model = VAE.FlowVAE(encoder_model, model_config.dim_z, decoder_model, model_config.style_architecture,
                               model_config.latent_flow_arch,
                               concat_midi_to_z0=model_config.concat_midi_to_z,  # FIXME deprecated
                               train_config=train_config)
    return encoder_model, decoder_model, ae_model


def build_extended_ae_model(model_config: ModelConfig, train_config: TrainConfig, idx_helper):
    """ Builds a spectral auto-encoder model, and a synth parameters regression model which takes
    latent vectors as input. Both models are integrated into an ExtendedAE model. """
    # Spectral VAE
    encoder_model, decoder_model, ae_model = build_ae_model(model_config, train_config)
    # Regression model - extension of the VAE model
    if model_config.params_regression_architecture.startswith("mlp_"):
        if not model_config.forward_controls_loss:
            raise AssertionError()   # Non-invertible MLP cannot inverse target values
        reg_arch = model_config.params_regression_architecture.replace("mlp_", "")
        reg_model = regression.MLPControlsRegression(reg_arch, model_config.dim_z, idx_helper,
                                                     cat_hardtanh_activation=model_config.params_reg_hardtanh_out,
                                                     cat_softmax_activation=model_config.params_reg_softmax,
                                                     model_config=model_config, train_config=train_config)
    elif model_config.params_regression_architecture.startswith("flow_"):
        if model_config.learnable_params_tensor_length <= 0:  # Flow models require dim_z to be equal to this length
            raise AssertionError()
        reg_arch = model_config.params_regression_architecture.replace("flow_", "")
        reg_model = regression.FlowControlsRegression(reg_arch, model_config.dim_z, idx_helper,
                                                      fast_forward_flow=model_config.forward_controls_loss,
                                                      cat_hardtanh_activation=model_config.params_reg_hardtanh_out,
                                                      cat_softmax_activation=model_config.params_reg_softmax,
                                                      model_config=model_config, train_config=train_config)
    else:
        raise NotImplementedError("Synth param regression arch '{}' not implemented"
                                  .format(model_config.params_regression_architecture))
    extended_ae_model = extendedAE.ExtendedAE(ae_model, reg_model)
    return encoder_model, decoder_model, ae_model, reg_model, extended_ae_model


def _is_attr_equal(attr1, attr2):
    """ Compares two config attributes - lists auto converted to tuples. """
    _attr1 = tuple(attr1) if isinstance(attr1, list) else attr1
    _attr2 = tuple(attr2) if isinstance(attr2, list) else attr2
    return _attr1 == _attr2


def check_configs_on_resume_from_checkpoint(new_model_config: ModelConfig, new_train_config: TrainConfig,
                                            config_json_checkpoint):
    """
    Performs a full consistency check between the last checkpoint saved config (stored into a .json file)
    and the new required config as described in config.py

    :raises: ValueError if any incompatibility is found

    :param new_model_config: model Class instance of the config.py file
    :param new_train_config: train Class instance of the config.py file
    :param config_json_checkpoint: config.py attributes from previous run, loaded from the .json file
    :return:
    """
    # Model config check TODO add/update attributes to check
    prev_config = config_json_checkpoint['model']
    attributes_to_check = ['name', 'run_name', 'encoder_architecture',
                           'dim_z', 'concat_midi_to_z', 'latent_flow_arch',
                           'logs_root_dir',
                           'note_duration',
                           # 'midi_notes',  # FIXME json 2D list to tuple conversion required for comparison
                           'stack_spectrograms', 'increased_dataset_size',
                           'stft_args', 'spectrogram_size', 'mel_bins']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_model_config.__dict__[attr]):
            raise ValueError("Model attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_model_config.__dict__[attr], prev_config[attr]))
    # Train config check TODO add.update attributes to check
    prev_config = config_json_checkpoint['train']
    attributes_to_check = ['minibatch_size', 'test_holdout_proportion', 'normalize_losses',
                           'optimizer', 'scheduler_name']
    for attr in attributes_to_check:
        if not _is_attr_equal(prev_config[attr], new_train_config.__dict__[attr]):
            raise ValueError("Train attribute '{}' is different in the new config.py ({}) and the old config.json ({})"
                             .format(attr, new_train_config.__dict__[attr], prev_config[attr]))


# Model Build tests - see also params_regression.ipynb
if __name__ == "__main__":

    import sys
    import pathlib  # Dirty path trick to import config.py from project root dir
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config

    # Manual config changes (for test purposes only)
    config.model.synth_params_count = 144

    _encoder_model, _decoder_model, _ae_model, _extended_ae_model \
        = build_extended_ae_model(config.model, config.train)
