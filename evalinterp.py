
from datetime import datetime

from evalconfig import InterpEvalConfig
from evaluation.load import ModelLoader
from evaluation.interpbase import NaivePresetInterpolation
from evaluation.interp import SynthPresetLatentInterpolation


def run(eval_config: InterpEvalConfig):

    t_start = datetime.now()
    n_actually_processed = 0

    assert eval_config.dataset_type == 'validation' or eval_config.dataset_type == 'test'
    model_loader = ModelLoader(
        eval_config.other_models[0]['base_model_path'], eval_config.device, eval_config.dataset_type)

    if eval_config.reference_model_path.name == 'LinearNaive':
        ref_preset_interpolator = NaivePresetInterpolation(
            model_loader.dataset, model_loader.dataset_type, model_loader.dataloader,
            eval_config.ref_model_interp_path, num_steps=eval_config.num_steps, u_curve='linear'
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
        model_loader = ModelLoader(m_config['base_model_path'], eval_config.device, eval_config.dataset_type)
        preset_interpolator = SynthPresetLatentInterpolation(
            model_loader, storage_path=m_config['interp_storage_path'],
            num_steps=eval_config.num_steps, u_curve=m_config['u_curve'], latent_interp=m_config['latent_interp'],
            reference_storage_path=eval_config.ref_model_interp_path, refine_level=m_config['refine_level']
        )
        preset_interpolator.use_reduced_dataset = eval_config.use_reduced_dataset
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


if __name__ == "__main__":
    run(InterpEvalConfig())
