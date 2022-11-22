import contextlib
import copy
import multiprocessing
import sys
import threading
import time
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Process

from evalconfig import InterpEvalConfig
from evaluation.load import ModelLoader
from evaluation.interpbase import NaivePresetInterpolation
from evaluation.interp import SynthPresetLatentInterpolation
import utils.text


def run(eval_config: InterpEvalConfig):
    """ Evaluates all models from the given InterpEvalConfig instance, including the reference one. """

    t_start = datetime.now()
    n_actually_processed = 0

    assert eval_config.dataset_type == 'validation' or eval_config.dataset_type == 'test'
    model_loader = ModelLoader(
        eval_config.other_models[0]['base_model_path'], eval_config.device, eval_config.dataset_type)

    if eval_config.reference_model_path.name == 'LinearNaive':
        ref_preset_interpolator = NaivePresetInterpolation(
            model_loader.dataset, model_loader.dataset_type, model_loader.dataloader,
            eval_config.ref_model_interp_path, num_steps=eval_config.num_steps, u_curve='linear',
            verbose=eval_config.verbose, verbose_postproc=eval_config.verbose_postproc
        )
    else:
        raise NotImplementedError()
    ref_preset_interpolator.use_reduced_dataset = eval_config.use_reduced_dataset
    was_processed = ref_preset_interpolator.try_process_dataset(
        (eval_config.ref_model_force_re_eval or eval_config.force_re_eval_all), eval_config.skip_audio_render)
    n_actually_processed += was_processed

    # Evaluate all "other" (i.e. non-reference) models
    for m_config in eval_config.other_models:
        print("\n\n\n\n--------------------------------------------------------------------------------------------")
        preset_interpolator = get_preset_interpolator(m_config, eval_config)
        try:
            force_re_eval = m_config['force_re_eval']
        except KeyError:  # Default: don't force re-evaluate
            force_re_eval = False
        force_re_eval = force_re_eval or eval_config.force_re_eval_all
        was_processed = preset_interpolator.try_process_dataset(force_re_eval, eval_config.skip_audio_render)
        n_actually_processed += was_processed

    duration_minutes = (datetime.now() - t_start).total_seconds() / 60.0
    print("\n\nFinished evaluation, {:.1f} min / model ({} models were actually evaluated, {:.1f} h total)".
          format((duration_minutes / n_actually_processed if n_actually_processed > 0 else -1.0),
                 n_actually_processed, duration_minutes / 60.0))


def get_preset_interpolator(m_config, eval_config: InterpEvalConfig):
    """
    :param m_config: configuration for a specific model to be evaluated
    :param eval_config: general eval config
    :return: The preset interpolator using the model described by m_config
    """
    model_loader = ModelLoader(m_config['base_model_path'], eval_config.device, eval_config.dataset_type)
    preset_interpolator = SynthPresetLatentInterpolation(
        model_loader, storage_path=m_config['interp_storage_path'],
        num_steps=eval_config.num_steps, u_curve=m_config['u_curve'], latent_interp=m_config['latent_interp'],
        reference_storage_path=eval_config.ref_model_interp_path, refine_level=m_config['refine_level'],
        verbose=eval_config.verbose, verbose_postproc=eval_config.verbose_postproc
    )
    preset_interpolator.use_reduced_dataset = eval_config.use_reduced_dataset
    return preset_interpolator


def eval_single_model(base_model_name: str, dataset_type='validation'):
    """
    Performs a parallel, non-blocking evaluation of the given model. Will erase any data from a previous eval.
    Is intended to be used after a training (for a queue of trainings runs) has finished.

    :param base_model_name: Relative path to the model, inside the logs dir (e.g. 'ablation/mlp3lBN_beta5.0e-06')
    """
    # Define a default eval config with a reference model but no actual model
    eval_config = InterpEvalConfig()
    eval_config.other_models = []
    eval_config.dataset_type = dataset_type
    eval_config.force_re_eval_all = True  # But we'll trigger a single-model eval only
    eval_config.skip_audio_render = False
    eval_config.use_reduced_dataset = False

    eval_config.verbose = True
    eval_config.verbose_postproc = False

    # Set the actual model
    eval_config.other_models = [{'base_model_name': base_model_name}, ]
    eval_config.set_default_config_values()  # both methods must be called after a model has been appended
    eval_config.build_models_storage_path()  # will set dataset type for the reference interpolation

    # Check that the reference model was already evaluated (test or validation)
    assert NaivePresetInterpolation.contains_eval_data(eval_config.ref_model_interp_path)

    if sys.gettrace() is None:  # No pycharm debugger attached
        ctx = multiprocessing.get_context('spawn')
        p = ctx.Process(target=_eval_single_model_process, args=(eval_config, True))
        p.start()
    else:  # Threaded (non-mp) debug mode
        t = threading.Thread(target=_eval_single_model_process, args=(eval_config, False))
        t.start()
    # No thread or process join absolutely needed: will be automatically joined when main process reaches its end


def _eval_single_model_process(eval_config: InterpEvalConfig, add_prefix_to_prints=False):
    assert len(eval_config.other_models) == 1
    m_config = eval_config.other_models[0]
    with redirect_stdout(utils.text.StdOutPrefixAdder("[[Evaluation: {}]] ".format(m_config['base_model_name'])
                                                      if add_prefix_to_prints else "")):
        preset_interpolator = get_preset_interpolator(m_config, eval_config)
        was_processed = preset_interpolator.try_process_dataset(True, eval_config.skip_audio_render)
        assert was_processed


if __name__ == "__main__":

    # Evaluate the current evalconfig.py (contains general configuration + reference model + other models)
    run(InterpEvalConfig())

    # OR evaluate a single model, e.g.:
    # eval_single_model('dev/autoeval_dlm2_betastart1.6e-8', dataset_type='validation')  # shouldn't use both evaluations

