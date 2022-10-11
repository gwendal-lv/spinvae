import abc
import copy
import datetime
from typing import List

import editdistance  # levenshtein edit distance
import numpy as np

from data.abstractbasedataset import AudioDataset
from data.dexeddataset import DexedDataset
from data.dataset import NsynthDataset


# TODO maybe use enums to go faster?
from data.surgedataset import SurgeDataset

_label_reference_strings = {
    # Pluck is another label
    # On a aussi les sons genre 'sitar' mais qui ne sont pas toujours pluck
    #
    'vocal': ['male', 'female', 'human', 'choir'],  # Human-like voices

    # Wind et brass sont a priori plutôt proches... ou pas ?
    'wind': ['wind', 'woodwind', 'clarinet', 'bassoon', 'oboe', 'flute'],

    'brass': ['brass', 'horn', 'trumpet', 'tuba', 'trombone', 'flugel', 'frenchhrn', 'saxophone'],

    'guitar': ['guitar', 'guit', 'banjo', 'clavinet', 'nylon', 'strat',  # gtr? (false positive risk)
               'clavichord', 'harpsichord'],  # clavichord is debatable; harpsichord actually uses guitar-like plectra

    'string': ['string', 'strng', 'violin', 'cello', 'viola', 'bowed'],  # Bowed or plucked strings (not guitar)

    'organ': ['organ', 'hammond', 'hmnd', 'leslie'],

    # Could be named "keys" or "e-piano". Add 'grand' ???
    'piano': ['piano', 'grandpno', 'gndpno', 'elecgrand', ' pno ', # 'pno' found very often... (false positive risk)
              'steinway',  'rhodes', 'rhds', 'wurlitzer'],     # will include 'epiano', ...

    'harmonic_perc': ['celesta', 'marimba', 'mrmba', 'xylo', 'glock', 'spiel', 'bell', 'tabla', 'vibraphone',
                      'mallet', 'kalimba', 'klmba', 'chimes', 'metallophone', 'timbales'],

    # Warning: clav peut être une percu ou un clavier.... ou clavecin/harpsichord
    # mallet? timbale?    tom? (false positive risk)  FIXME move some to harmonic_perc
    'percussive': ['perc', 'drum', 'snare', 'cymbal', 'claves', 'clap', 'conga', 'bongo', 'gong',
                   'tamborine', 'tambourine', 'cow bell', 'dingle', 'hi hat'],

    'sfx': ['effect', 'drone', 'snd efx', 'chopper'],  # Laser is excluded, too many false positives

    # Timbre descriptors, not related to an instrument: warm, bright, dark(/growl), (FX?)....
    # warm/dark/bright are assigned to very few UIDs (120/60/40), most of them being false positive
    # 'warm': ['warm'],  # hollow?
    # 'dark': ['dark', 'night', 'devil', 'massacre'],  # ....  TODO use language model to get the closest words?
    # 'bright': ['bright'],  # metal? glass?

    # Typical digital 'instrument types' -> also found in nsynth and/or surge datasets
    'bass': ['bass', 'fretless'],  # means a bass guitar or a bass synth... fretless?? also included in guitar
    'pad': ['pad'],
    'lead': ['lead'],
    'pluck': ['pluck', 'pizz'],  # pizz?
}

# If a preset's name exactly contains one of those strings, the associated label won't be assigned to the preset
_label_rejection_strings = {
    'vocal': ['chorus', 'chord', 'chirp', 'organ', 'org', 'echo', 'stg choir', 'chor',
              'huang', 'horn', 'perc', 'clav', 'piano', 'malet', 'mallet', 'keys'],
    'guitar': ['clarinet', 'state', 'stretch', 'abstract', 'static',
               'strange', 'station', 'orchestra'],  # 'strat' leads to many false positives
    'piano': [],
    'harmonic_perc': ['table'],
    'percussive': ['super', 'hyper', 'clavi', 'octave', 'bingo', 'claver', 'pad', 'table'],
    'sfx': ['perfect', 'layer', 'laer', 'claver', 'effectsz'],
    'wind': ['clavinet', 'wild', 'flugel', 'rewind', 'windows', 'flutter'],
    'brass': ['bass', 'chorus', 'chor', 'flute'],
    'string': ['steinway', 'strange', 'loop',    # 'string' close to steinw(ay)
               'miscella', 'spring'],
    'organ': ['harmonica', 'formant', 'gregorian'],  # 'hammond' close to harmoni(ca)
    'warm': ['warp', 'ward', 'ware', 'wars'],
    'dark': ['trk', 'standard'],  # 'sndtrk' comes close to 'dark'
    'bright': [],
    'bass': ['brass', 'drum'],
    'pad': [],
    'lead': ['lady'],
    'pluck': [],
}

_instrument_label_keys = [
    'vocal', 'wind', 'brass', 'guitar', 'string', 'organ', 'piano', 'harmonic_perc', 'percussive', 'sfx',
    'bass', 'pad', 'lead', 'pluck'
]
for _label in _instrument_label_keys:
    if not _label in list(_label_reference_strings.keys()):
        raise ValueError("Instrument label '{}' cannot be found in _label_reference_strings dictionary."
                         .format(_label))

# Convert NSynth's instrument families into out labels
_nsynth_label_to_instrument_labels = {
    'bass': ['bass'], 'brass': ['brass'], 'flute': ['wind'], 'guitar': ['guitar'], 'keyboard': ['piano'],
    'mallet': ['harmonic_perc'],  # TODO add 'percussion' too?
    'organ': ['organ'],
    'reed': ['wind'],   # TODO add 'brass' too?
    'string': ['string'], 'synth_lead': ['lead'], 'vocal': ['vocal']
}
# Convert Surge's instrument categories into our labels
_surge_category_to_instrument_labels = {  # Ordered by number of patch per category (most represented first)
    'Leads': ['lead'],
    'Basses': ['bass'], 'Bass': ['bass'],
    'Pads': ['pad'], 'Pad': ['pad'],  # also: atmosphere, ambiance, soundscapes?
    'Plucks': ['pluck'],
    'Keys': ['piano'],
    'FX': ['sfx'],
    # Polysynths, Rhythms, Sequences, Synths: hard to label... (some presets will be labeled by name)
    'Atmospheres': ['pad', 'sfx'],  # sfx is quite often but not always appropriate
    'Percussion': ['percussive'],  # sometimes also contains harmonic percussions: kalimba, bells, ...
    'Drums': ['percussive'],
    'Winds': ['wind'],
    'Ambiance': ['pad'],
    'Soundscapes': ['pad', 'sfx'],  # sometimes 'pad' or 'sfx', not both of them...
    'Brass': ['brass'],
    'Organs': ['organ'],
    'Vox': ['vocal'],
    'Mallets': ['harmonic_perc'],
    'Bells': ['harmonic_perc'],
    'Strings': ['string'],
    'Drones': ['sfx']
}



class LabelerABC(abc.ABC):
    def __init__(self, ds: AudioDataset):
        """ Abstract Base Class for any Labeling class. Does not contain the methods that actually assign labels. """
        self.ds = ds
        # str labels, per UID
        self.labels_per_UID = {item_UID: list() for item_UID in self.ds.valid_preset_UIDs}
        # Lists of UIDs which have a given label
        self.label_dataset_UIDs = {k: list() for k in _label_reference_strings.keys()}
        # List of UIDs which have no label
        self.UIDs_with_no_label = list()

    def __str__(self):
        num_UIDs_with_label = len(self.ds.valid_preset_UIDs) - len(self.UIDs_with_no_label)
        return "Labeler for the {} dataset. {}/{} UIDs have a label ({:.1f} %)" \
            .format(self.ds.synth_name, num_UIDs_with_label,
                    len(self.ds.valid_preset_UIDs), 100.0 * num_UIDs_with_label / len(self.ds.valid_preset_UIDs))

    @property
    def all_available_labels(self):
        """ Returns the list of all available labels. """
        return sorted(list(_label_reference_strings.keys()))

    @property
    def instrument_labels(self):
        """ Returns the list of labels related to musical instrument (e.g. 'voice', 'piano', 'guitar', 'perc', ...) """
        return sorted(_instrument_label_keys)

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

    @abc.abstractmethod
    def extract_labels_for_UID(self, item_UID: int):
        pass

    def extract_labels_from_name(self, name: str):
        """ Tries to extract a label from a given string, using sub-strings and the Levenshtein edit distance. """
        label_assigned = {k: False for k in _label_reference_strings.keys()}
        for label in _label_reference_strings.keys():
            for ref_str in _label_reference_strings[label]:
                # Get smaller substring, and edit distance
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



class NameBasedLabeler(LabelerABC):
    def __init__(self, ds: AudioDataset):
        """ Class for assigning labels based on synth preset names only
        (no proper labels provided in the original dataset). """
        super().__init__(ds)

    def extract_labels_for_UID(self, item_UID: int):
        """ :returns: A Dict of booleans indicating whether each label is assigned to the given item_UID. """

        # - - - Pre-process the name - - -
        # We want a case-insensitive distance, don't care about head/tail spaces
        name = self.ds.get_name_from_preset_UID(item_UID).strip().lower()
        # TODO remove numbers and special chars?
        # e.g. "B_R_A_S_S  " has an obvious label
        label_assigned = self.extract_labels_from_name(name)

        # Also get a second name, if available (e.g. concat cartridge name at the beginning)
        if isinstance(self.ds, DexedDataset):
            cartridge_name = self.ds.get_cartridge_name_from_preset_UID(item_UID).lower().strip()
            cartridge_label_assigned = self.extract_labels_from_name(cartridge_name)
            for label, assigned in label_assigned.items():
                label_assigned[label] = (assigned or cartridge_label_assigned[label])  # OR operation on labels
        return label_assigned



class NSynthReLabeler(LabelerABC):
    def __init__(self, ds: NsynthDataset):
        """ Specific re-labeler for NSynth (which already contains labelled sounds). """
        super().__init__(ds)
        self.ds = ds  # Redundant assignation (already done in super ctor), for auto-completion

    def extract_labels_for_UID(self, item_UID: int):
        """ Re-assigns the existing NSynth instrument_family labels. """
        # We don't use preset names (uninformative, contains the family and an index only)
        label_assigned = {k: False for k in _label_reference_strings.keys()}
        instrument_family = self.ds.get_original_instrument_family(item_UID)
        for label in _label_reference_strings.keys():
            if label in _nsynth_label_to_instrument_labels[instrument_family]:
                label_assigned[label] = True
        return label_assigned



class SurgeReLabeler(LabelerABC):
    def __init__(self, ds: SurgeDataset):
        """ Specific re-labeler for NSynth (which already contains labelled sounds). """
        super().__init__(ds)
        self.ds = ds  # Redundant assignation (already done in super ctor), for auto-completion

    def extract_labels_for_UID(self, item_UID: int):
        """ Assigns labels to UIDs, first using the Surge category, then using preset names themselves. """
        label_assigned = {k: False for k in _label_reference_strings.keys()}
        instrument_family = self.ds.get_original_instrument_family(item_UID)
        try:
            for label in _label_reference_strings.keys():
                # first, get an explicit label from the category
                if label in _surge_category_to_instrument_labels[instrument_family]:
                    label_assigned[label] = True
        except KeyError:  # some instrument families cannot be classified (don't exist in the _surge labels dict)
            pass
        # then, try to get a second label from the preset's name
        preset_name = self.ds.get_name_from_preset_UID(item_UID).lower().strip()
        second_label_assigned = self.extract_labels_from_name(preset_name)
        for label, assigned in label_assigned.items():
            label_assigned[label] = (assigned or second_label_assigned[label])  # OR operation on labels
        return label_assigned



if __name__ == "__main__":

    # Run as main: for DEBUG purposes only

    import data.dataset
    ds_type = 'dexed'

    dataset_kwargs = {
        'note_duration': (3.0, 1.0), 'n_fft': 512, 'fft_hop': 256, 'Fs': 16000,
        'midi_notes': ((41, 75), (48, 75), (56, 75), (63, 75), (56, 25), (56, 127)),
        'multichannel_stacked_spectrograms': True, 'data_storage_root_path': "/media/gwendal/Data/Datasets"
    }

    if ds_type == "dexed":
        _ds = data.dataset.DexedDataset(
            ** dataset_kwargs,
            vst_params_learned_as_categorical='all', continuous_params_max_resolution=50
        )
        labeler = NameBasedLabeler(_ds)
    elif ds_type == "surge":
        _ds = data.dataset.SurgeDataset(
            ** dataset_kwargs
        )
        labeler = SurgeReLabeler(_ds)
    elif ds_type == 'nsynth':
        _ds = data.dataset.NsynthDataset(
            **dataset_kwargs,
            exclude_instruments_with_missing_notes=False,
        )
        labeler = NSynthReLabeler(_ds)
    else:
        raise AssertionError()

    labeler.extract_labels(verbose=True)

    print(labeler)
    labeler.display_samples_per_label()

    # 1-by-1 debug tests
    _labels = labeler.extract_labels_for_UID(81549)
