"""
Extension of the dexed.py file - because it's very large already
"""

from itertools import permutations

import numpy as np

# TODO automatic test of all of those permutations (check that no algo is found 2 times, etc...)


# Symmetric permutations of oscillators, for each algorithm, considering there is feedback
_osc_permutations_per_algo_with_feedback \
    = {1:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       2:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       3:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       4:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       5:  np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 5 - - -
                       [3, 4, 1, 2, 5, 6]]),  # swap 1st and 2nd branches
       6:  np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 6 - - -
                       [3, 4, 1, 2, 5, 6]]),  # swap 1st and 2nd branches
       7:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       8:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       9:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       10: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 10 - - -
                       [1, 2, 3, 4, 6, 5]]),  # swap osc 5 and 6 (mods of osc 4)
       11: np.asarray([[1, 2, 3, 4, 5, 6]]),
       12: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 12 - - -
                       [1, 2, 3, 4, 6, 5],   # swap osc 4, 5, 6  (3! perms)
                       [1, 2, 3, 5, 4, 6],
                       [1, 2, 3, 5, 6, 4],
                       [1, 2, 3, 6, 4, 5],
                       [1, 2, 3, 6, 5, 4]]),
       13: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 13 - - -
                       [1, 2, 3, 5, 4, 6]]),  # swap osc 4, 5
       14: np.asarray([[1, 2, 3, 4, 5, 6]]),
       15: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 15 - - -
                       [1, 2, 3, 4, 6, 5]]),  # swap osc 5, 6
       16: np.asarray([[1, 2, 3, 4, 5, 6]]),
       17: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 17 - - -
                       [1, 2, 5, 6, 3, 4]]),  # swap leaf branches 3-4 and 5-6
       18: np.asarray([[1, 2, 3, 4, 5, 6]]),
       19: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 19 - - -
                       [1, 2, 3, 5, 4, 6]]),  # swap osc 4, 5
       20: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 20 - - -
                       [1, 2, 3, 4, 6, 5],  # swap 1-2 and 5-6 osc pairs
                       [2, 1, 3, 4, 5, 6],
                       [2, 1, 3, 4, 6, 5]]),
       21: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 21 - - -
                       [1, 2, 3, 5, 4, 6],  # swap 1-2 and 4-5 osc pairs
                       [2, 1, 3, 4, 5, 6],
                       [2, 1, 3, 5, 4, 6]]),
       22: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 22 - - -
                       [1, 2, 3, 5, 4, 6],   # swap osc 3, 4, 5
                       [1, 2, 4, 3, 5, 6],
                       [1, 2, 4, 5, 3, 6],
                       [1, 2, 5, 3, 4, 6],
                       [1, 2, 5, 4, 3, 6]]),
       23: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 23 - - -
                       [1, 2, 3, 5, 4, 6]]),  # swap osc 4, 5
       24: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 24 - - -
                       [1, 2, 3, 5, 4, 6],   # swap osc groups 1-2 and 3-4-5
                       [1, 2, 4, 3, 5, 6],
                       [1, 2, 4, 5, 3, 6],
                       [1, 2, 5, 3, 4, 6],
                       [1, 2, 5, 4, 3, 6],
                       [2, 1, 3, 4, 5, 6],
                       [2, 1, 3, 5, 4, 6],
                       [2, 1, 4, 3, 5, 6],
                       [2, 1, 4, 5, 3, 6],
                       [2, 1, 5, 3, 4, 6],
                       [2, 1, 5, 4, 3, 6]]),
       25: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 25 - - -
                       [1, 2, 3, 5, 4, 6],   # swap osc groups 1-2-3 and 4-5
                       [1, 3, 2, 4, 5, 6],
                       [1, 3, 2, 5, 4, 6],
                       [2, 1, 3, 4, 5, 6],
                       [2, 1, 3, 5, 4, 6],
                       [2, 3, 1, 4, 5, 6],
                       [2, 3, 1, 5, 4, 6],
                       [3, 1, 2, 4, 5, 6],
                       [3, 1, 2, 5, 4, 6],
                       [3, 2, 1, 4, 5, 6],
                       [3, 2, 1, 5, 4, 6]]),
       26: np.asarray([[1, 2, 3, 4, 5, 6]]),
       27: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 27 - - -
                       [1, 2, 3, 4, 6, 5]]),  # swap osc 5, 6
       28: np.asarray([[1, 2, 3, 4, 5, 6]]),
       29: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 29 - - -
                       [2, 1, 3, 4, 5, 6]]),  # swap osc 1, 2
       30: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 30 - - -
                       [1, 6, 3, 4, 5, 2],  # swap osc 1, 2, 6
                       [2, 1, 3, 4, 5, 6],
                       [2, 6, 3, 4, 5, 1],
                       [6, 1, 3, 4, 5, 2],
                       [6, 2, 3, 4, 5, 1]]),
       }
# permutations for algos 31 and 32: build automatically (resp. 24 and 120 permutations...)
perms = np.zeros((24, 6), dtype=int)  # Algo 31: swap 1, 2, 3, 4
perms[:, 0:4] = np.asarray(list(set(permutations([1, 2, 3, 4]))))
perms[:, 4] = 5
perms[:, 5] = 6
_osc_permutations_per_algo_with_feedback[31] = perms
perms = np.zeros((120, 6), dtype=int)  # Algo 31: swap 1, 2, 3, 4, 5
perms[:, 0:5] = np.asarray(list(set(permutations([1, 2, 3, 4, 5]))))
perms[:, 5] = 6
_osc_permutations_per_algo_with_feedback[32] = perms


# TODO build array of symmetries when there is no feedback, for a single algo (do not consider the others yet)
_osc_permutations_per_algo_without_feedback \
    = {1:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       2:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       3:  np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 3 (and 4, see below) - - -
                       [4, 5, 6, 1, 2, 3]]),  # swap branches 3-2-1 and 6-5-4
       5:  np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 5 (and 6, see below) - - -
                       [1, 2, 5, 6, 3, 4],   # swap branches 2-1, 4-3 and 5-6
                       [3, 4, 1, 2, 5, 6],
                       [3, 4, 5, 6, 1, 2],
                       [5, 6, 1, 2, 3, 4],
                       [5, 6, 3, 4, 1, 2]]),
       7:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       8:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       9:  np.asarray([[1, 2, 3, 4, 5, 6]]),
       10: _osc_permutations_per_algo_with_feedback[10],  # Algo 10: same symmetry, with or without feedback
       11: _osc_permutations_per_algo_with_feedback[10],  # Without feedback: algo 11 becomes algo 10
       12: _osc_permutations_per_algo_with_feedback[12],  # Algo 12: same symmetry, with or without feedback
       13: _osc_permutations_per_algo_with_feedback[12],  # Without feedback: algo 13 becomes algo 12
       14: _osc_permutations_per_algo_with_feedback[15],  # Without feedback: algo 14 becomes algo 15 (see next line)
       15: _osc_permutations_per_algo_with_feedback[15],  # Algo 15: same symmetry, with or without feedback
       16: _osc_permutations_per_algo_with_feedback[17],  # Without feedback: algo 16 becomes algo 17 (see next line)
       17: _osc_permutations_per_algo_with_feedback[17],  # Algo 17: same symmetry, with or without feedback
       18: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 18 - - -
                       [1, 3, 2, 4, 5, 6]]),  # swap osc 2, 3
       19: _osc_permutations_per_algo_with_feedback[19],  # Algo 19: same symmetry, with or without feedback
       20: _osc_permutations_per_algo_with_feedback[20],  # Algo 20: same symmetry, with or without feedback
       21: _osc_permutations_per_algo_with_feedback[21],  # Algo 21: same symmetry, with or without feedback
       22: _osc_permutations_per_algo_with_feedback[22],  # Algo 22: same symmetry, with or without feedback
       23: _osc_permutations_per_algo_with_feedback[23],  # Algo 23: same symmetry, with or without feedback
       24: _osc_permutations_per_algo_with_feedback[24],  # Algo 24: same symmetry, with or without feedback
       25: _osc_permutations_per_algo_with_feedback[25],  # Algo 25: same symmetry, with or without feedback
       26: _osc_permutations_per_algo_with_feedback[27],  # Without feedback: algo 26 becomes algo 27 (see next line)
       27: _osc_permutations_per_algo_with_feedback[27],  # Algo 27: same symmetry, with or without feedback
       28: np.asarray([[1, 2, 3, 4, 5, 6]]),
       29: np.asarray([[1, 2, 3, 4, 5, 6],   # - - - Symmetric algorithm 29 - - -
                       [1, 2, 5, 6, 3, 4],   # swap 1, 2 and groups 4-3 and 6-5
                       [2, 1, 3, 4, 5, 6],
                       [2, 1, 5, 6, 3, 4]]),
       30: _osc_permutations_per_algo_with_feedback[30],  # Algo 30: same symmetry, with or without feedback
       31: _osc_permutations_per_algo_with_feedback[31],  # Algo 31: same symmetry, with or without feedback
       32: np.asarray(list(set(permutations([1, 2, 3, 4, 5, 6]))))  # swap all
       }
_osc_permutations_per_algo_without_feedback[4] = _osc_permutations_per_algo_without_feedback[3]
_osc_permutations_per_algo_without_feedback[6] = _osc_permutations_per_algo_without_feedback[5]


# TODO build dict of symmetric osc+algos when there is no feedback. Each element is a list of algos (including itself)
# an algo can become identical to other algos, such that all of their combined permutations are symmetries
_identical_algos_without_feedback \
    = {1: [1, 2], 2: [1, 2],  # Algos 1 and 2
       3: [3, 4], 4: [3, 4],  # Algos 3 and 4
       5: [5, 6], 6: [5, 6],  # Algos 5 and 6
       7: [7, 8, 9], 8: [7, 8, 9], 9: [7, 8, 9],  # Algos 7, 8, 9
       10: [10, 11], 11: [10, 11],  # Algos 10 and 11
       12: [12, 13], 13: [12, 13],  # Algos 12 and 13
       14: [14, 15], 15: [14, 15],  # Algos 14 and 15
       16: [16, 17], 17: [16, 17],  # Algos 16 and 17
       # Algos 18 to 25 are unique (as long as all oscillators have a non-zero volume)
       18: [18], 19: [19], 20: [20], 21: [21], 22: [22], 23: [23], 24: [24], 25: [25],
       26: [26, 27], 27: [26, 27],  # Algos 26 and 27
       # Algos 28 to 32 are unique (as long as all oscillators have a non-zero volume)
       28: [28], 29: [29], 30: [30], 31: [31], 32: [32]
       }
_algo_and_osc_permutations_without_feedback = dict()
for _algo in range(1, 33):
    _other_algos = _identical_algos_without_feedback[_algo]
    _algo_and_osc_permutations_without_feedback[_algo] \
        = (np.hstack([np.ones((_osc_permutations_per_algo_without_feedback[_other_alg].shape[0], ), dtype=int)
                      * _other_alg for _other_alg in _other_algos]),
           np.vstack([_osc_permutations_per_algo_without_feedback[_other_alg] for _other_alg in _other_algos])
           )


def get_algorithms_and_oscillators_permutations(algo: int, feedback: bool):
    """ TODO doc

    https://scsynth.org/uploads/default/optimized/1X/983d97a124e5b4c61890bb4c7b1454ef2ef0f012_2_1023x544.jpeg
    """
    # TODO try to optimize this (after it works properly...)
    # To prevent errors, algos and operators indexes will be translated to [1, 32] instead of [0, 31]
    algo += 1

    if feedback:  # If there is feedback, only the original algorithm is possible
        osc_permutations = _osc_permutations_per_algo_with_feedback[algo]
        algo_permutations = np.ones((len(osc_permutations), ), dtype=int) * algo
    else:  # If there's no feedback, there may be many more symmetries
        algo_permutations, osc_permutations = _algo_and_osc_permutations_without_feedback[algo]

    # remove 1 to all indexes, to return actual indexes
    algo_permutations = np.asarray(algo_permutations, dtype=int) - 1
    osc_permutations = np.asarray(osc_permutations, dtype=int) - 1
    return algo_permutations, osc_permutations

