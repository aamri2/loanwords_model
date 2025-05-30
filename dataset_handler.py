from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk
import datasets
from typing import Optional, Union
from collections.abc import Callable
from transformers import Wav2Vec2FeatureExtractor


def prepare_targets():
    """Extracts TIMIT syllables into a dataset, which is prepared for training."""
    timit = load_dataset('timit_asr', data_dir='../timit/TIMIT')
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in batch['phonetic_detail']['utterance']]
        # positions and labels of all C+VC+ sequences (excluding non-target diphthongs)
        target_indices = [i for i, phone in enumerate(utterance) if phone in target_labels.keys()]
        vowel_indices = [i for i, phone in enumerate(utterance) if phone in timit_vowels]
        target_start = [None for i in target_indices]
        target_end = [None for i in target_indices]
        target_utterance = [target_labels[utterance[i]] for i in target_indices]
        for i, target_index in enumerate(target_indices):
            preceding_vowel_index = max(target_indices[:i] + [j for j in vowel_indices if j < target_index] + [0])
            target_start[i] = batch['phonetic_detail']['start'][preceding_vowel_index + 1]
            following_vowel_index = min([j for j in target_indices[i:] if j > target_index] + [j for j in vowel_indices if j > target_index] + [len(utterance)])
            target_end[i] = batch['phonetic_detail']['stop'][following_vowel_index - 1]
        batch['target_start'] = target_start
        batch['target_end'] = target_end
        batch['target_utterance'] = target_utterance    
        return batch
    timit = timit.map(find_targets)

    def generator(batch):
        for row in range(len(batch)):
            sampling_rate = batch[row]['audio']['sampling_rate']
            for target_start, target_end, target_utterance in zip(batch[row]['target_start'], batch[row]['target_end'], batch[row]['target_utterance']):
                yield {
                    'audio_array': batch[row]['audio']['array'][target_start:target_end],
                    'audio_sampling_rate': sampling_rate,
                    'target_utterance': target_utterance
                }
            
    targets_test = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['test']}, split = datasets.Split.TEST)
    targets_train = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['train']}, split = datasets.Split.TRAIN)
    targets = datasets.DatasetDict({'test': targets_test, 'train': targets_train})
    targets.save_to_disk('../timit_targets')

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        batch['input_values'] = feature_extractor(batch['audio_array'], sampling_rate=batch['audio_sampling_rate']).input_values[0]
        batch['labels'] = batch['target_utterance']
        return batch

    prepared_targets = targets.map(prepare_dataset, remove_columns=targets.column_names['train'])
    prepared_targets = prepared_targets.class_encode_column('labels')

    return prepared_targets

def prepare_masked_targets():
    """
    Prepares TIMIT utterances, with additional columns 'sample_start' and 'sample_stop'.
    """
    timit = load_dataset('timit_asr', data_dir='../timit/TIMIT')
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in batch['phonetic_detail']['utterance']]
        # positions and labels of all C+VC+ sequences (excluding non-target diphthongs)
        target_indices = [i for i, phone in enumerate(utterance) if phone in target_labels.keys()]
        vowel_indices = [i for i, phone in enumerate(utterance) if phone in timit_vowels]
        target_start = [None for i in target_indices]
        target_end = [None for i in target_indices]
        target_utterance = [target_labels[utterance[i]] for i in target_indices]
        for i, target_index in enumerate(target_indices):
            preceding_vowel_index = max(target_indices[:i] + [j for j in vowel_indices if j < target_index] + [0])
            target_start[i] = batch['phonetic_detail']['start'][preceding_vowel_index + 1]
            following_vowel_index = min([j for j in target_indices[i:] if j > target_index] + [j for j in vowel_indices if j > target_index] + [len(utterance)])
            target_end[i] = batch['phonetic_detail']['stop'][following_vowel_index - 1]
        batch['target_start'] = target_start
        batch['target_end'] = target_end
        batch['target_utterance'] = target_utterance    
        return batch
    timit = timit.map(find_targets)

    def generator(batch):
        for row in range(len(batch)):
            audio_array = batch[row]['audio']['array']
            sampling_rate = batch[row]['audio']['sampling_rate']
            for target_start, target_end, target_utterance in zip(batch[row]['target_start'], batch[row]['target_end'], batch[row]['target_utterance']):
                yield {
                    'audio_array': audio_array,
                    'audio_sampling_rate': sampling_rate,
                    'target_utterance': target_utterance,
                    'target_start': target_start,
                    'target_end': target_end
                }

    
    targets_test = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['test']}, split = datasets.Split.TEST)
    targets_train = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['train']}, split = datasets.Split.TRAIN)
    targets = datasets.DatasetDict({'test': targets_test, 'train': targets_train})
    targets.save_to_disk('../timit_masked_targets')

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        batch['input_values'] = feature_extractor(batch['audio_array'], sampling_rate=batch['audio_sampling_rate']).input_values[0]
        batch['labels'] = batch['target_utterance']
        batch['sample_start'] = batch['target_start']
        batch['sample_stop'] = batch['target_end']
        return batch

    masked_targets = targets.map(prepare_dataset, remove_columns=targets.column_names['train'])
    masked_targets = masked_targets.class_encode_column('labels')

    return masked_targets

# class DatasetLoader:
#     """A wrapper for info needed to dynamically load and prepare a dataset or multiple datasets.

#     Attributes:
#         name: A string or list of strings
#     """

#     def __init__(self, name: Union[str, List[str]], data_dir: Optional[str] = None, from_disk = False):
#         self.name = name
#         self.from_disk = True


def prepare(dataset: Union[Dataset, DatasetDict], input: Union[str, list[str]], output: Union[str, list[str]]) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """Prepares a Dataset for training.

    Given a Dataset, it prepares the desired inputs and outputs
    in a valid format for training.

    Args:
        dataset: The dataset to prepare.
        input: The column containing the input values. A list of
            strings is treated as a nested column.
        output: The column containing the output values. A list
            of strings is treated as a nested column.
        TODO:
        input_map: A function applied to input values...
    
    Returns:
        A prepared dataset with two columns: 'input_values' and
        'labels'.
    """

    if isinstance(input, str):
        input = [input]
    
    if isinstance(output, str):
        output = [output]

    columns_to_remove: list[str] = [column_name for column_name in dataset.column_names if column_name not in input[0] and column_name not in output[0]]
    dataset = dataset.remove_columns(columns_to_remove)

    return dataset

