import copy
import datetime
from typing import List

import editdistance  # levenshtein edit distance
import numpy as np

from data.abstractbasedataset import AudioDataset


# TODO maybe use enums to go faster?
_label_reference_strings = {
    # Pluck is another label
    # On a aussi les sons genre 'sitar' mais qui ne sont pas toujours pluck
    #
    'voice': ['male', 'female', 'choir'],  # Human-like voices

    # Wind et brass sont a priori plutôt proches... ou pas ?
    'wind': ['wind', 'woodwind', 'clarinet', 'bassoon', 'oboe', 'flute'],

    'brass': ['brass', 'horn', 'trumpet', 'tuba', 'trombone', 'flugel', 'frenchhrn'],

    'guitar': ['guitar', 'guit', 'banjo', 'clavinet', 'fretless', 'nylon'],  # gtr ???

    'string': ['string', 'strng', 'violin', 'cello', 'viola'],  # Bowed or plucked strings (not guitar)

    'organ': ['organ', 'hammond'],

    # Could be named "keys" or "e-piano". Add 'grand' ???
    'piano': ['piano', 'grandpno', 'gndpno', 'steinway',  # 'pno' found very often... (false positive risk)
              'rhodes', 'wurlitzer'],  # will include 'epiano', ... 'pno' vu assez souvent aussi
    # Faux positifs: 'lunarTides'

    # marimba et xylophone ??? proches piano et percu ?
    'harmonic_perc': ['celesta', 'marimba', 'xylo', 'glock', 'spiel'],

    # Warning: clav peut être une percu ou un clavier.... ou clavecin/harpsichord
    # mallet? timbale?    tom? (false positive risk)
    'percussive': ['perc', 'bell', 'drum', 'snare', 'tabla', 'claves', 'clap', 'conga', 'bongo',
                   'tamborine', 'mallet'],


    # Descripteurs de timbre, pas liés à un instrument: pad, warm, bright, dark(/growl), (FX?)....
    # warm/dark/bright are assigned to very few UIDs (120/60/40), most of them being false positive
    # 'warm': ['warm'],  # hollow?
    # 'dark': ['dark', 'night', 'devil', 'massacre'],  # ....  TODO use language model to get the closest words?
    # 'bright': ['bright'],  # metal? glass?
    'pad': ['pad'],
    'lead': ['lead'],
    'pluck': ['pluck'],  # pizz?
}
# If the name exactly contains one of those strings, the associated label won't be assigned to the label
_label_rejection_strings = {
    'voice': ['chorus', 'chord'],
    'guitar': ['clarinet'],  # prevent
    'piano': [],
    'harmonic_perc': [],
    'percussive': ['super', 'hyper', 'clavi'],  # 'clavinet' conflicts with clave
    'wind': ['clavinet', 'wild', 'flugel', 'rewind'],
    'brass': ['bass', 'chorus', 'chor', 'flute'],
    'string': ['steinway'],  # 'string' close to steinw(ay)
    'organ': ['harmonica'],  # 'hammond' close to harmoni(ca)
    'warm': ['warp', 'ward', 'ware', 'wars'],
    'dark': ['trk', 'standard'],  # 'sndtrk' comes close to 'dark'
    'bright': [],
    'pad': [],
    'lead': ['lady'],
    'pluck': [],
}


# TODO ajouter "second name" ou qqchose comme ça, pour analyser aussi le nom de cartridge

class NameBasedLabeler:
    def __init__(self, ds: AudioDataset):
        self.ds = ds
        # str labels, per UID
        self.labels_per_UID = {item_UID: list() for item_UID in self.ds.valid_preset_UIDs}
        # Lists of UIDs which have a given label
        self.label_dataset_UIDs = {k: list() for k in _label_reference_strings.keys()}
        # List of UIDs which have no label
        self.UIDs_with_no_label = list()

    def __str__(self):
        num_UIDs_with_label = len(self.ds.valid_preset_UIDs) - len(self.UIDs_with_no_label)
        return "Labeler for the {} dataset. {}/{} UIDs have a label ({:.1f} %)"\
            .format(self.ds.synth_name, num_UIDs_with_label,
                    len(self.ds.valid_preset_UIDs), 100.0 * num_UIDs_with_label / len(self.ds.valid_preset_UIDs))

    def extract_labels(self, verbose=False):
        """ Extracts labels from the whole dataset. """
        t_start = datetime.datetime.now()
        for item_UID in self.ds.valid_preset_UIDs:
            label_assigned = self.extract_labels_for_UID(item_UID)
            # check if assigned or not
            unassigned = True
            for label, assigned in label_assigned.items():
                if assigned:
                    self.label_dataset_UIDs[label].append(item_UID)
                    self.labels_per_UID[item_UID].append(label)
                    unassigned = False
            if unassigned:
                self.UIDs_with_no_label.append(item_UID)
        if verbose:
            delta_t = (datetime.datetime.now() - t_start).total_seconds()
            print("Labels extracted in {:.1f}s ({:.1f}ms/item)"
                  .format(delta_t, 1000.0 * delta_t / len(self.ds.valid_preset_UIDs)))

    def extract_labels_for_UID(self, item_UID):
        """ :returns: A Dict of booleans indicating whether each label is assigned to the given item_UID. """
        label_assigned = {k: False for k in _label_reference_strings.keys()}

        # - - - Pre-process the name - - -
        # We want a case-insensitive distance, don't care about head/tail spaces
        name = self.ds.get_name_from_preset_UID(item_UID).strip().lower()
        # TODO also get a second name, if available (e.g. concat cartridge name at the beginning)

        # TODO remove numbers and special chars?
        # e.g. "B_R_A_S_S  " has an obvious label

        for label in _label_reference_strings.keys():
            for ref_str in _label_reference_strings[label]:

                # Get smaller substring, and edit distance (print result???)
                substr_idx = name.find(ref_str[0])
                if substr_idx >= 0:  # -1 indicates that char was not found
                    edit_distance_th = len(ref_str) // 5  # FIXME EDIT DISTANCE THRESHOLD
                    # First try: we get a substr that has the same length as the ref_str
                    name_substr = name[substr_idx:substr_idx + len(ref_str)]
                    small_edit_distance = (editdistance.eval(ref_str, name_substr) <= edit_distance_th)
                    if not small_edit_distance:
                        # Second try: we get a substr that starts ** and ends ** with the same letter as the ref_str
                        # E.g.: 'orgn0' edit distance vs. 'organ' is 2, but 'orgn' edit distance vs. 'organ' is 1
                        substr_end_idx = name_substr.rfind(ref_str[-1])
                        if substr_end_idx >= 0:
                            smaller_name_substr = name_substr[0:substr_end_idx+1]
                            small_edit_distance = (editdistance.eval(ref_str, smaller_name_substr) <= edit_distance_th)
                else:
                    small_edit_distance = False

                # Label is automatically assigned if the name contains an exact reference string (NOT fully redundant
                # with edit distance)
                #     or if the edit distance (vs. cropped name) is smaller than a threshold
                if ref_str in name or small_edit_distance:
                    # rejections: edit distance threshold leads to many false positives...
                    reject = any([(rejection_str in name) for rejection_str in _label_rejection_strings[label]])

                    if not reject:
                        label_assigned[label] = True
                        break  # don't need to check other reference strings - continue to the next label
        return label_assigned

    def get_subset_UIDs_and_names(self, _UIDs: List[int], subset_len=10):
        """ Returns a limited amount of UID and names associated to the given label
            (for debug/display purposes). """
        shuffled_UIDs = copy.deepcopy(_UIDs)
        np.random.shuffle(shuffled_UIDs)  # In-place shuffling
        return {_UID: self.ds.get_name_from_preset_UID(_UID, long_name=True) for _UID in shuffled_UIDs[0:subset_len]}

    def display_samples_per_label(self):
        """ Display a few random UIDs and names for each label. """
        for label, UIDs in self.label_dataset_UIDs.items():
            print("\n--------> Label '{}' has {} UIDs:".format(label, len(UIDs)))
            print(self.get_subset_UIDs_and_names(UIDs))

        print("\n--------> {} UIDs with no label:".format(len(self.UIDs_with_no_label)))
        print(self.get_subset_UIDs_and_names(self.UIDs_with_no_label, subset_len=100))



if __name__ == "__main__":


    # TODO prévoir labels pour surge et nsynth également...


    import data.dataset
    _ds = data.dataset.DexedDataset(
        note_duration=(3.0, 1.0),
        n_fft=512,
        fft_hop=256,
        Fs=16000,
        midi_notes=((41, 75), (48, 75), (56, 75), (63, 75), (56, 25), (56, 127)),
        multichannel_stacked_spectrograms=True,
        data_storage_root_path="/media/gwendal/Data/Datasets",

        vst_params_learned_as_categorical='all',
        continuous_params_max_resolution=50
    )

    labeler = NameBasedLabeler(_ds)
    labeler.extract_labels(verbose=True)

    print(labeler)
    labeler.display_samples_per_label()

    # 1-by-1 debug tests
    _labels = labeler.extract_labels_for_UID(81549)
