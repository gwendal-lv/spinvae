"""
Base module for the Dexed, which can be imported into any other module with very few dependencies.
The dexed.py module enables sound rendering but requires librenderman to be available on the machine.
"""
from typing import List

import numpy as np

import synth.dexedpermutations


class DexedCharacteristics:
    """
    Class with static methods only, to retrieve constant characteristics about the DX7/Dexed synthesizer.
    E.g., it describes which parameters correspond to which indices, which ones can be permuted, ....
    """
    def __init__(self):
        pass

    @staticmethod
    def get_midi_key_related_param_indexes():
        """ Returns a list of indexes of all DX7 parameters that apply a modulation depending on the MIDI key
        (note and/or velocity). These will be very hard to learn without providing multiple-notes input
        to the encoding network. """
        # (6. 'OSC KEY SYNC' (LFO) does NOT depend on the midi note (it syncs or not LFO phase on midi event).)
        # All the KEY L/R stuff (with breakpoint at some MIDI note) effects are really dependant on the MIDI key.
        # 36. breakpoint. Values 0 to 99 correspond to MIDI notes 9 to 108 (A-1 to C8)
        # 37/38: L/R scale (curve) depth (-> EG level scaling only?)
        # 39/40: L/R scale (=curve) type: +/-lin or +/-exp. (-> EG level scaling only?)
        # 41: rate scaling (-> EG rate scaling, longer decay for bass notes)
        # 43: key velocity (-> general OP amplitude increases(?) with MIDI velocity)
        return sorted([(36 + 22*i) for i in range(6)]\
            + [(37 + 22*i) for i in range(6)] + [(38 + 22*i) for i in range(6)]\
            + [(39 + 22*i) for i in range(6)] + [(40 + 22*i) for i in range(6)] \
            + [(41 + 22 * i) for i in range(6)] + [(43 + 22 * i) for i in range(6)])

    @staticmethod
    def get_mod_wheel_related_param_indexes():
        """ Returns a list of indexes of all DX7 parameters that influence sound depending on the MIDI
        mod wheel. These should always be learned because they are also related to LFO modulation
        (see https://fr.yamaha.com/files/download/other_assets/9/333979/DX7E1.pdf page 26) """
        # OPx A MOD SENS + Pitch general mod sens
        return [(42 + 22*i) for i in range(6)] + [14]

    @staticmethod
    def get_param_cardinality(param_index):
        """ Returns the number of possible values for a given parameter. """
        if param_index == 4:  # Algorithm
            return 32
        elif param_index == 5:  # Feedback
            return 8
        elif param_index == 6:  # OSC key sync (off/on)
            return 2
        elif param_index == 11:  # LFO key sync (off/on)
            return 2
        elif param_index == 12:  # LFO wave
            return 6
        elif param_index == 14:  # pitch modulation sensitivity
            return 8
        elif param_index >= 23:  # oscillators (operators) params
            if (param_index % 22) == (32 % 22):  # OPx Mode (ratio/fixed)
                return 2
            elif (param_index % 22) == (33 % 22):  # OPx F coarse
                return 32
            elif (param_index % 22) == (35 % 22):  # OPx OSC Detune
                return 15  # -7 to +7  (15 steps, including central position)
            elif (param_index % 22) == (39 % 22):  # OPx L Key Scale (-lin, -exp, +exp, +lin)
                return 4
            elif (param_index % 22) == (40 % 22):  # OPx R Key Scale (-lin, -exp, +exp, +lin)
                return 4
            elif (param_index % 22) == (41 % 22):  # OPx Rate Scaling
                return 8
            elif (param_index % 22) == (42 % 22):  # OPx A modulation sensitivity
                return 4
            elif (param_index % 22) == (43 % 22):  # OPx Key Velocity
                return 8
            elif (param_index % 22) == (44 % 22):  # OPx Switch (off/on)
                return 2
            else:  # all other are 'continuous' but truly present 100 steps
                return 100
        else:
            return 100

    @staticmethod
    def get_param_types(operator_index=False):
        """ Returns a list of strings describing each parameter. If operator_index is False, all operator
        will be considered to have the same types if parameters (no position information). """
        param_types = \
            ['Cutoff', 'Resonance', 'Output', 'MASTER_TUNE_ADJ', 'ALGORITHM', 'FEEDBACK', 'OSC_KEY_SYNC', 'LFO_SPEED',
             'LFO_DELAY', 'LFO_PM_DEPTH', 'LFO_AM_DEPTH', 'LFO_KEY_SYNC', 'LFO_WAVE', 'MIDDLE_C', 'P_MODE_SENS',
             'PITCH_EG_RATE_1', 'PITCH_EG_RATE_2', 'PITCH_EG_RATE_3', 'PITCH_EG_RATE_4',
             'PITCH_EG_LEVEL_1', 'PITCH_EG_LEVEL_2', 'PITCH_EG_LEVEL_3', 'PITCH_EG_LEVEL_4',]
        op_param_types = [
            'OPx_EG_RATE_1', 'OPx_EG_RATE_2', 'OPx_EG_RATE_3', 'OPx_EG_RATE_4',
            'OPx_EG_LEVEL_1', 'OPx_EG_LEVEL_2', 'OPx_EG_LEVEL_3', 'OPx_EG_LEVEL_4',
            'OPx_OUTPUT_LEVEL', 'OPx_MODE', 'OPx_F_COARSE', 'OPx_F_FINE', 'OPx_OSC_DETUNE',
            'OPx_BREAK_POINT', 'OPx_L_SCALE_DEP', 'OPx_R_SCALE_DEP', 'OPx_L_KEY_SCALE', 'OPx_R_KEY_SCALE',
            'OPx_RATE_SCALING', 'OPx_A_MOD_SENS', 'OPx_KEY_VELOCITY', 'OPx_SWITCH'
        ]
        if not operator_index:
            return param_types + op_param_types * 6
        else:
            all_ops_param_types = list()
            for i in range(6):
                for p_type in op_param_types:
                    all_ops_param_types.append(p_type.replace('OPx', 'OP{}'.format(i+1)))
            return param_types + all_ops_param_types

    @staticmethod
    def get_numerical_params_indexes():
        indexes = [0, 1, 2, 3, 5,  # cutoff, reso, output, master tune, feedback (card:8)
                   7, 8, 9, 10,  # lfo speed, lfo delay (before LFO actually modulates), lfo pm depth, lfo am depth
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # transpose, pitch mod sensitivity, pitch EG rates/levels
        for i in range(6):  # operators
            for j in [23, 24, 25, 26, 27, 28, 29, 30]:  # rates and levels
                indexes.append(j + 22*i)
            indexes.append(31 + 22*i)  # output level
            indexes.append(33 + 22*i)  # freq coarse
            indexes.append(34 + 22*i)  # freq fine
            indexes.append(35 + 22*i)  # detune (these 3 parameters kind of overlap...)
            indexes.append(36 + 22*i)  # L/R scales breakpoint
            indexes.append(37 + 22*i)  # L scale depth
            indexes.append(38 + 22*i)  # R scale depth
            indexes.append(41 + 22*i)  # rate scaling (card:8)
            indexes.append(42 + 22*i)  # amplitude mod sensitivity (card:4)
            indexes.append(43 + 22*i)  # key velocity (card:8)
        return indexes

    @staticmethod
    def get_categorical_params_indexes():
        indexes = [4, 6, 11, 12]  # algorithm, osc key sync, lfo key sync, lfo wave
        for i in range(6):  # operators
            indexes.append(32 + 22*i)  # mode (ratio or fixed frequency)
            indexes.append(39 + 22*i)  # L scale
            indexes.append(40 + 22*i)  # R scale
            indexes.append(44 + 22*i)  # op on/off switch
        return indexes

    @staticmethod
    def get_op_output_level_indices():
        return [31 + 22*i for i in range(6)]

    @staticmethod
    def get_L_R_scale_indices():
        return [39 + 22*i for i in range(6)] + [40 + 22*i for i in range(6)]

    @staticmethod
    def get_operators_params_indexes_groups() -> List[range]:
        """ Returns a list of 6 ranges, where each range contains the indices of all parameters of an operator. """
        return [range(23 + 22 * i, 23 + 22 * (i+1)) for i in range(6)]

    @staticmethod
    def get_algorithms_and_oscillators_permutations(algo: int, feedback: bool):
        return synth.dexedpermutations.get_algorithms_and_oscillators_permutations(algo, feedback)

    @staticmethod
    def get_similar_preset(preset: np.ndarray, variation: int, learnable_indices: List[int], random_seed=0):
        """ Data augmentation method: returns a slightly modified preset, which is (hopefully) quite similar
            to the input preset. """
        if variation == 0:
            return preset
        rng = np.random.default_rng((random_seed + 987654321 * variation))
        # First: change algorithm to a similar one
        preset = synth.dexedpermutations.change_algorithm_to_similar(preset, variation, random_seed)
        # Then: change a few learned parameters if variation > 1: algorithm is the hardest parameter to learn,
        # so we provide a special data augmentation for this parameter only.
        if variation == 1:
            return preset
        # If variation >= 2: random noise applied to most of parameters
        cat_indexes = DexedCharacteristics.get_categorical_params_indexes()
        # don't choose a limited subset of param to augment, but use a <1.0 noise std
        for idx in learnable_indices:
            if idx == 4:  # algorithm
                pass
            # cat params: depends
            elif idx in cat_indexes:
                card = DexedCharacteristics.get_param_cardinality(idx)
                # General cat params: key sync, lfo sync, lfo wave: 100% randomization
                if idx in [6, 11, 12]:
                    preset[idx] = rng.integers(0, card) / (card - 1.0)
                # OP mode: quite risky to change it... (likely to lead to inaudible/unlikely sounds)
                elif idx > 32 and ((idx - 32) % 22 == 0):
                    pass
                # L/R scales: invert lin/exp (keep +/- sign)
                elif idx in DexedCharacteristics.get_L_R_scale_indices():
                    if preset[idx] < 0.5:
                        preset[idx] = rng.integers(0, 1, endpoint=True) / 3
                    else:
                        preset[idx] = rng.integers(2, 3, endpoint=True) / 3
            # "continuous" (discrete ordinal) params: triangle or gaussian noise, std depends on cardinality
            else:
                card = DexedCharacteristics.get_param_cardinality(idx)
                if card < 100:  # Discrete ordinal params with a few values: smaller probability of change
                    noise = rng.choice([-1, 0, 1], p=[0.05, 0.9, 0.05]) / (card - 1.0)
                    lol = 0
                else:
                    # Noise with 1 discrete increment of 0.5 standard deviation
                    noise = rng.normal(0.0, 0.5 / (card - 1.0))
                    # Very small volume values: only positive noise
                    if idx in DexedCharacteristics.get_op_output_level_indices() and preset[idx] < 0.1:
                        noise = np.abs(noise)
                preset[idx] += noise
        return np.clip(preset, 0.0, 1.0)

