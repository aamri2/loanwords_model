import torch
from spec import TrainingDatasetSpec, _SEPARATOR, TestDatasetSpec
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_from_disk
import datasets
import json
import pandas as pd
from typing import Sequence, overload, cast, Mapping, Callable
import phonemizer.separator
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PhonemeCTCTokenizer
import phonemizer
from pyacoustics.speech_filters import speech_shaped_noise
import numpy as np
from torchaudio.functional import forced_align, merge_tokens
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from tqdm import tqdm
from functools import cached_property
from collections import defaultdict
import re

_DATASET_PATH = '../'
_DATASET_PREFIX = 'prep'

class TrainingDataset():
    """
    Uses TrainingDatasetSpec to parse and load datasets.
    """

    spec: TrainingDatasetSpec
    path: str
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict

    def __init__(self, spec: str | TrainingDatasetSpec, path: str | None = None):
        self.spec = TrainingDatasetSpec(spec)
        self.path = path if path else f'{_DATASET_PATH}{_DATASET_PREFIX}{_SEPARATOR}{spec}'
        self.dataset = self.load_dataset()
    
    @property
    def vocab_path(self) -> str:
        return self.path + '/vocab.json'
    
    @cached_property
    def label2id(self) -> dict[str, int]:
        return self.get_dataset_vocab()
    
    @cached_property
    def id2label(self) -> dict[int, str]:
        return {v: k for k, v in self.label2id.items()}

    @cached_property
    def consonants(self) -> list[str]:
        return get_consonants(self.spec)
    
    @cached_property
    def vowels(self) -> list[str]:
        return get_vowels(self.spec)
    
    def get_translator(self, other_spec: str | TrainingDatasetSpec | TestDatasetSpec) -> dict[str, str]:
        """Returns a translator from the dataset into the transcription system of another."""

        return translators[str(self.spec.family)][str(other_spec)]

    def load_dataset(self) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
        """Loads a dataset given its specification."""
        
        return datasets.load_from_disk(self.path)

    def get_dataset_vocab(self) -> dict[str, int]:
        """Loads a dataset's vocabulary given its specification."""

        with open(self.vocab_path, encoding = 'utf-8') as f:
            vocab = json.load(f)
        return vocab
    
    def get_split(self, splits: str | Sequence[str]):
        """Return the split or concatenated splits of the dataset."""

        if not isinstance(splits, str) and isinstance(splits, Sequence):
            return datasets.concatenate_datasets([cast(datasets.DatasetDict, self.dataset)[split] for split in splits])
        else:
            return cast(datasets.DatasetDict, self.dataset)[splits]
        
    def add_files(self, test_dataset):
        """Adds file names from a test dataset in a column {test_dataset}_file."""

        test_df = test_dataset.dataset.to_pandas()
        input_values = test_df['input_values'].apply(tuple)
        if isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({
                split: self.dataset[split].add_column(f'{test_dataset.spec}_file', self.dataset[split].to_pandas().apply(lambda row: next(iter(test_df['file'][input_values == tuple(row['input_values'])]), None), axis=1))
                for split in self.dataset.keys()
            })
        else:
            self.dataset = self.dataset.add_column(f'{test_dataset.spec}_file', self.dataset.to_pandas().apply(lambda row: next(iter(test_df['file'][input_values == tuple(row['input_values'])]), None), axis=1))

class TrainingDatasetMap(Mapping):
    """Mapping for training datasets."""

    def __init__(self, map=()):
        self._map = dict(map)

    @overload
    def __getitem__(self, key: str | TrainingDatasetSpec) -> TrainingDataset: ...
    @overload
    def __getitem__(self, key: Sequence[str | TrainingDatasetSpec]) -> tuple[TrainingDataset, ...]: ...
    def __getitem__(self, key: str | TrainingDatasetSpec | Sequence[str | TrainingDatasetSpec]) -> TrainingDataset | tuple[TrainingDataset, ...]:
        if not (isinstance(key, (str, TrainingDatasetSpec)) or (isinstance(key, Sequence) and all(isinstance(k, (str, TrainingDatasetSpec)) for k in key))):
            raise TypeError('Probability specification must be a string or list of strings!')
        
        if isinstance(key, Sequence) and not isinstance(key, str):
            return tuple(self.__getitem__(k) for k in key)
        
        key = str(key)
        if key not in self._map.keys():
            return self.__missing__(key)
        
        return self._map[key]

    def __missing__(self, key: str):
        try:
            self._map[key] = TrainingDataset(key)
            return self._map[key]
        except FileNotFoundError:
            raise NotImplementedError(f'Cannot dynamically create training dataset {key}.')
    
    def __iter__(self):
        return iter(self._map)
    
    def __len__(self):
        return len(self._map)
    
d = TrainingDatasetMap()


class TranslationDict(dict):
    """A dictionary that defaults to returning the key."""

    def __missing__(self, key):
        return key

consonants = {
    'timit': ['b', 'ch', 'd', 'dh', 'dx', 'f', 'g', 'jh', 'k', 'hh', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh'],
    'timitEC': ['p', 't', 'k', 'b', 'd', 'ɡ', 'h', 'f', 'θ', 's', 'ʃ', 'v', 'ð', 'z', 'ʒ', 'tʃ', 'dʒ', 'm', 'n', 'ŋ', 'r', 'l', 'j', 'w', 'y'],
    'librispeech': ["b", "d", "dʒ", "f", "h", "j", "k", "l", "m", "n", "p", "s", "t", "tʃ", "v", "w", "z", "ð", "ŋ", "ɡ", "ɹ", "ɾ", "ʃ", "ʒ", "θ"],
    'librispeechFR': ['b', 'd', 'dʒ', 'f', 'j', 'k', 'l', 'm', 'n', 'p', 's', 't', 'tʃ', 'v', 'w', 'z', 'ɡ', 'ɲ', 'ʁ', 'ʃ', 'ʒ'],
    'bl': ['n', 'b', 'k', 's', 'Z', 'v', 'j', 'm', 'w', 'g', 't', 'R', 'l', 'd', 'S', 'N', 'z', 'p', 'f']
}

vowels = {
    'timit': ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ah', 'ow', 'uh', 'uw'],
    'timitEC': ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix'],
    'bl': ['i', 'y', 'e', 'E', 'deux', 'neuf', 'a', 'a~', 'o', 'O', 'o~', 'u', 'U~'],
}

translators = defaultdict(lambda: defaultdict(TranslationDict), {
    'timit': defaultdict(TranslationDict, {
        'wv': TranslationDict({
            'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ',
            'sh': 'ʃ', 'g': 'ɡ', 'ch': 'tʃ', 'dh': 'ð', 'dx': 'd', 'jh': 'dʒ', 'ng': 'ŋ', 'th': 'θ', 'y': 'j', 'zh': 'ʒ',
        }),
        'cc': TranslationDict({
            'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ',
            'sh': 'ʃ', 'g': 'ɡ', 'ch': 'tʃ', 'dh': 'ð', 'dx': 'd', 'jh': 'dʒ', 'ng': 'ŋ', 'th': 'θ', 'y': 'j', 'zh': 'ʒ', 'hh': 'h',
        }),
    }),
    'bl': defaultdict(TranslationDict, {
        'wv': TranslationDict({
            'i': 'i', 'y': 'y', 'e': 'e', 'E': '\u025b', 'deux': '\u00f8', 'neuf': '\u0153', 'a': 'a', 'a~': '\u0251\u0303', 'o': 'o', 'O': '\u0254', 'o~': '\u0254\u0303', 'u': 'u', 'U~': '\u025b\u0303',
            'g': 'ɡ', 'S': 'ʃ', 'R': 'r'
        }),
    }),
    'cc': defaultdict(TranslationDict, {
        'wv': TranslationDict({
            'sh': 'ʃ', 'g': 'ɡ', 'ch': 'tʃ', 'dh': 'ð', 'dx': 'd', 'dj': 'dʒ', 'ng': 'ŋ', 'th': 'θ', 'y': 'j', 'zh': 'ʒ',
        }),
    }),
})


def get_consonants(spec: str | TrainingDatasetSpec) -> list[str]:
    """Given a specification, gets the consonants in the transcription system of the dataset."""

    spec = TrainingDatasetSpec(spec)
    return consonants[spec.family]

def get_vowels(spec: str | TrainingDatasetSpec) -> list[str]:
    """Given a specification, gets the vowels in the transcription system of the dataset."""

    spec = TrainingDatasetSpec(spec)
    return vowels[spec.family]

def make_noisy(dataset, snr):
    """Takes a prepared dataset and returns it with noise added."""

    def add_noise(batch):
        audios = [np.array(input_values) for input_values in batch['input_values']]
        all_audio = np.hstack(audios)
        noise = speech_shaped_noise._noise_from_signal(all_audio)
        for i, audio in enumerate(audios):
            level = speech_shaped_noise._dbspl(audio)
            audios[i] = speech_shaped_noise._mix_noise(audio, noise, level, snr)[1].tolist()
        return {'input_values': audios}
    
    dataset = dataset.map(add_noise, batched=True, batch_size=64)
    return dataset

def align_ctc(dataset, model):
    """
    Given a prepared dataset and a CTC model trained on that dataset, gives alignments.
    
    Alignments are given in the columns "start" and "end" of the returned dataset.
    """

    def get_alignments(batch):
        input_values = torch.tensor([batch['input_values']])
        with torch.no_grad():
            log_probs = model(input_values).logits
        targets = torch.tensor([batch['labels']])
        tokens, scores = forced_align(log_probs, targets, blank=model.config.pad_token_id)
        merged_tokens = merge_tokens(tokens[0], scores[0], blank=model.config.pad_token_id)
        start, end = zip(*[(token_span.start, token_span.end) for token_span in merged_tokens])
        batch['start'] = start
        batch['end'] = end
        return batch

    dataset = dataset.map(get_alignments)
    return dataset

def random_substrings(aligned_dataset, samples_to_frames_ratio, substring_length = None, substring_ratio = 0.3):
    """
    Takes a prepared dataset with alignments and returns randomly selected substrings.

    Args:
        aligned_dataset: A prepared dataset with additional
            columns 'start' and 'end' containing frame-alignments.
        samples_to_frames_ratio: The number of samples per frame, where alignment
            is at the frame-level.
        substring_length: Takes precedence over 'substring_ratio'. The length
            of the desired substrings, expressed in number of tokens.
        substring_ratio: The length of the desired substrings, expressed
            as a ratio of the total number of tokens.
    """

    def use_substring_length(string_length):
        return torch.tensor([substring_length]).int()

    def use_substring_ratio(string_length):
        rounding = torch.randint(0, 2, (1,))
        return torch.tensor([string_length * substring_ratio]).int() + rounding
    
    get_substring_length = use_substring_length if substring_length else use_substring_ratio

    def get_substrings(batch):
        n_tokens = get_substring_length(len(batch['labels']))
        start_token = torch.randint(1, len(batch['labels']) - n_tokens, (1,))
        end_token = start_token + n_tokens
        # substring is taken from end of previous token to start of following token
        start_frame = batch['end'][start_token - 1]
        end_frame = batch['start'][end_token]
        audio_substring = batch['input_values'][(np.floor(start_frame * samples_to_frames_ratio - 1).astype('int')):(np.ceil(end_frame * samples_to_frames_ratio).astype('int'))]
        labels_substring = batch['labels'][start_token:end_token]
        batch['input_values'] = audio_substring
        batch['labels'] = labels_substring
        return batch
    
    aligned_dataset = aligned_dataset.map(get_substrings, remove_columns = ['start', 'end'])
    return aligned_dataset

def noisy_and_substrings(aligned_dataset, snr, samples_to_frames_ratio, substring_length = None, substring_ratio = 0.3, noisy_ratio = 0.5, substring_subset_ratio = 0.5, keep_originals = False):
    """Helper function makes noisy and substring split of aligned dataset."""

    if isinstance(aligned_dataset, DatasetDict):
        new_dataset = DatasetDict()
        for key, dataset in aligned_dataset.items():
            new_dataset[key] = noisy_and_substrings(dataset, snr, samples_to_frames_ratio, substring_length, substring_ratio, noisy_ratio, substring_subset_ratio, keep_originals)
        return new_dataset
    
    random_indices = np.random.choice(range(len(aligned_dataset)), size=(len(aligned_dataset,)), replace=False)

    # make noisy subset
    if noisy_ratio > 0:
        noisy_dataset = make_noisy(
            aligned_dataset.select(random_indices[:int(noisy_ratio*len(aligned_dataset))]), snr
        )
    else:
        noisy_dataset = Dataset.from_dict({'input_values': [], 'labels': [], 'start': [], 'end': []})
    if keep_originals:
        clean_dataset = aligned_dataset
    else:
        clean_dataset = aligned_dataset.select(random_indices[int(noisy_ratio*len(aligned_dataset)):])

    # make substring subsets
    if substring_subset_ratio > 0:
        noisy_substring_dataset = random_substrings(
            noisy_dataset.select(range(int(substring_subset_ratio*len(noisy_dataset)))), samples_to_frames_ratio, substring_length, substring_ratio
        )
        clean_substring_dataset = random_substrings(
            clean_dataset.select(range(int(substring_subset_ratio*len(clean_dataset)))), samples_to_frames_ratio, substring_length, substring_ratio
        )
    else:
        noisy_substring_dataset = Dataset.from_dict({'input_values': [], 'labels': []})
        clean_substring_dataset = Dataset.from_dict({'input_values': [], 'labels': []})
    if keep_originals:
        noisy_string_dataset = noisy_dataset.remove_columns(['start', 'end'])
        clean_string_dataset = clean_dataset.remove_columns(['start', 'end'])
    else:
        noisy_string_dataset = noisy_dataset.select(range(int(substring_subset_ratio*len(noisy_dataset)), len(noisy_dataset))).remove_columns(['start', 'end'])
        clean_string_dataset = clean_dataset.select(range(int(substring_subset_ratio*len(clean_dataset)), len(clean_dataset))).remove_columns(['start', 'end'])

    new_dataset = datasets.concatenate_datasets([noisy_substring_dataset, noisy_string_dataset, clean_substring_dataset, clean_string_dataset])
    new_dataset = new_dataset.shuffle().flatten_indices()
    return new_dataset

def throughPretrainedModel(dataset, model):
    """Runs a prepared dataset through a given base model."""

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def process_row(batch):
        input_values = feature_extractor(batch['input_values'], sampling_rate=16000, return_tensors='pt', padding=True)['input_values']
        with torch.no_grad():
            batch['input_values'] = model(input_values).last_hidden_state.detach()
        return batch
        
    return dataset.map(process_row, batched=True, batch_size=8)

def prepare_timitMV_w2v2():
    """Runs TIMIT MV through w2v2."""

    try:
        timit = load_from_disk('../prep_timitMV')
    except FileNotFoundError:
        timit = prepare_masked_targets()
        timit.save_to_disk('../prep_timitMV')

    model = Wav2Vec2Model.from_pretrained('../w2v2')
    
    return throughPretrainedModel(timit, model)

def process_wv_responses(wv_responses: pd.DataFrame) -> Dataset:
    """
    Takes a subset of the WV responses, and annotates them with
    the columns 'audio', 'file', 'label', 'vowel', 'language',
    and 'CVC' in a HF Dataset for further processing.
    """

    wv_responses_ds = Dataset.from_dict({
        'audio': [f'../stimuli_world_vowels/{file}.wav' for file in wv_responses['filename']],
        'file': wv_responses['filename'],
        'label': wv_responses['assimilation'],
        'vowel': wv_responses['#phone'],
        'language': wv_responses['language_stimuli'],
        'CVC': wv_responses['prev_phone'] + wv_responses['#phone'] + wv_responses['next_phone'],
    })\
        .cast_column('audio', datasets.Audio())\
        .class_encode_column('file')\
        .class_encode_column('label')\
        .class_encode_column('vowel')\
        .class_encode_column('language')\
        .class_encode_column('CVC')\
        .shuffle()
        
    return wv_responses_ds

@overload
def unannotate_dataset(dataset: Dataset) -> Dataset: ...
@overload
def unannotate_dataset(dataset: DatasetDict) -> DatasetDict: ...
def unannotate_dataset(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    """Remove all columns except 'input_values' and 'label(s)'."""

    if isinstance(dataset, DatasetDict):
        columns = sum(dataset.column_names.values(), start=list()) 
    else:
        columns = dataset.column_names
    columns_to_remove = set(column for column in columns if column not in ['input_values', 'label', 'labels'])
    prepared_dataset = dataset.remove_columns(list(columns_to_remove))
    return prepared_dataset

def k_fold(annotated_dataset: Dataset, k=10, stratify_by_column='vowel', split_by_column: str|None='file', distribute_CVCs=True) -> DatasetDict:
    """Splits an annotated dataset into 10 folds."""

    splits = {}

    if split_by_column:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True)
        for i, (train_dev_split, test_split) in enumerate(sgkf.split(X = np.arange(len(annotated_dataset)), y = annotated_dataset[stratify_by_column], groups = annotated_dataset[split_by_column])):
            splits[f'train_{i}'], splits[f'dev_{i}'] = (annotated_dataset.select(train_dev_split).select(split) for split in train_test_split(np.arange(len(train_dev_split)), stratify = annotated_dataset.select(train_dev_split)[stratify_by_column], test_size = 0.1))
            splits[f'test_{i}'] = annotated_dataset.select(test_split)
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        for i, (train_dev_split, test_split) in enumerate(skf.split(X = np.arange(len(annotated_dataset)), y = annotated_dataset[stratify_by_column])):
            splits[f'train_{i}'], splits[f'dev_{i}'] = (annotated_dataset.select(train_dev_split).select(split) for split in train_test_split(np.arange(len(train_dev_split)), stratify = annotated_dataset.select(train_dev_split)[stratify_by_column], test_size = 0.1))
            splits[f'test_{i}'] = annotated_dataset.select(test_split)

    if distribute_CVCs:
        # ensure CVCs in test set always appear in train set
        folds_to_validate = k 
        while folds_to_validate > 0:
            folds_to_validate = k
            for i in range(k):
                missing_cvcs = set(splits[f'test_{i}']['CVC']).difference(splits[f'train_{i}']['CVC']) # in test set but not train
                if len(set(splits[f'test_{i}'].filter(lambda row: row['CVC'] in missing_cvcs)['file'])) == len(missing_cvcs): # no missing CVCs, or only in one file each
                    folds_to_validate -= 1 # empty pass through all folds means distributed
                    continue
                for cvc in missing_cvcs:
                    stratified_column = splits[f'test_{i}'].filter(lambda row: row['CVC'] == cvc)[stratify_by_column][0] # same CVC means same vowel
                    if cvc in splits[f'dev_{i}']['CVC']: # can just swap from dev set
                        file_to_swap_train = splits[f'train_{i}'].filter(lambda row: # filtering swapping candidates
                            row[stratify_by_column] == stratified_column
                            and ( # either in another file in train, or not in test
                                row['CVC'] in splits[f'train_{i}'].filter(lambda row_2: row_2['file'] != row['file'])['CVC']
                                or row['CVC'] not in splits[f'test_{i}']['CVC']
                            )
                        ).shuffle()[0]['file']
                        train_swap_dataset = splits[f'train_{i}'].filter(lambda row: row['file'] == file_to_swap_train)
                        file_to_swap_dev = splits[f'dev_{i}'].filter(lambda row: row['CVC'] == cvc).shuffle()[0]['file']
                        dev_swap_dataset = splits[f'dev_{i}'].filter(lambda row: row['file'] == file_to_swap_dev)
                        # swap files
                        splits[f'train_{i}'] = datasets.concatenate_datasets([
                            splits[f'train_{i}'].filter(lambda row: row['file'] != file_to_swap_train),
                            dev_swap_dataset
                        ])
                        splits[f'dev_{i}'] = datasets.concatenate_datasets([
                            splits[f'dev_{i}'].filter(lambda row: row['file'] != file_to_swap_dev),
                            train_swap_dataset
                        ])
    
    return DatasetDict(splits)

def prepare_wvResponses():
    """Prepares World Vowels stimuli with individual responses as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[human_responses['language_indiv'] == 'english']

    wvResponses = process_wv_responses(human_responses)
    wvResponses_split = wvResponses.train_test_split(0.2, 0.8, stratify_by_column = 'vowel_language')
    wvResponses_test_dev = wvResponses_split['test'].train_test_split(0.5, 0.5, stratify_by_column = 'vowel_language')
    wvResponses = DatasetDict({'train': wvResponses_split['train'], 'test': wvResponses_test_dev['train'], 'dev': wvResponses_test_dev['test']})

    prep_wvResponses = audio_to_input_values(wvResponses)
    return unannotate_dataset(prep_wvResponses)

def prepare_wvENNonnativeResponses10Fold():
    """WV responses by English speakers on all stimuli."""

    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'english') & (human_responses['language_stimuli'] != 'EN')]
    wvENNonnativeResponses = process_wv_responses(human_responses)

    wvENNonnativeResponses = wvENNonnativeResponses.map(lambda batch: {'vowel_language': f"{batch['vowel']}_{batch['language']}"})
    prep_wvENNonnativeResponses = audio_to_input_values(wvENNonnativeResponses)
    return unannotate_dataset(k_fold(prep_wvENNonnativeResponses, stratify_by_column='vowel_language', distribute_CVCs = False))

def prepare_wvFRNonnativeResponses10Fold():
    """WV responses by French speakers on all stimuli."""

    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'french') & (human_responses['language_stimuli'] != 'FR')]

    wvFRNonnativeResponses = process_wv_responses(human_responses)
    wvFRNonnativeResponses = wvFRNonnativeResponses.map(lambda batch: {'vowel_language': f"{batch['vowel']}_{batch['language']}"})
    prep_wvFRNonnativeResponses = audio_to_input_values(wvFRNonnativeResponses)
    return unannotate_dataset(k_fold(prep_wvFRNonnativeResponses, stratify_by_column='vowel_language', distribute_CVCs = False))

def prepare_wvENResponses():
    """Prepares World Vowels stimuli with individual responses as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'english') * (human_responses['language_stimuli'] == 'EN')]

    wvENResponses = process_wv_responses(human_responses)
    prep_wvENResponses = audio_to_input_values(wvENResponses)
    return unannotate_dataset(prep_wvENResponses)

def prepare_wvENResponses10Fold():
    """Prepares World Vowels stimuli with individual responses as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'english') * (human_responses['language_stimuli'] == 'EN')]

    wvENResponses = process_wv_responses(human_responses)
    prep_wvENResponses = audio_to_input_values(wvENResponses)
    return unannotate_dataset(k_fold(prep_wvENResponses))

def prepare_wvFRResponses10Fold():
    """Prepares French World Vowels stimuli with individual responses (from French-speaking participants) as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'french') * (human_responses['language_stimuli'] == 'FR')]

    wvFRResponses = process_wv_responses(human_responses)
    prep_wvFRResponses = audio_to_input_values(wvFRResponses)
    return unannotate_dataset(k_fold(prep_wvFRResponses))

def prepare_wvEN():
    """Prepares World Vowels English stimuli."""
    
    wv = pd.read_csv('WorldVowels_stimuli.csv')
    files = ['../wv/' + i for i in wv[wv['language'] == 'EN']['#file_extract']]
    phone = list(wv[wv['language'] == 'EN']['#phone'])
    wvEN = Dataset.from_dict({'audio': files, 'labels': phone}, split=datasets.Split.TRAIN)\
        .cast_column('audio', datasets.Audio())\
        .class_encode_column('labels')
    prep_wvEN = audio_to_input_values(wvEN)
    return unannotate_dataset(prep_wvEN)

def prepare_wvEN10Fold():
    "Prepares WV English stimuli with vowel labels for 10-fold cross validation."
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'english') * (human_responses['language_stimuli'] == 'EN')]
    human_responses = human_responses.groupby('filename').first().reset_index() # one row per file

    wvEN = process_wv_responses(human_responses)
    wvEN = wvEN.remove_columns('label').rename_column('vowel', 'label')
    prep_wvEN = audio_to_input_values(wvEN)
    prep_wvEN10Fold = k_fold(prep_wvEN, stratify_by_column='label')
    return unannotate_dataset(prep_wvEN10Fold)

def prepare_wvFR10Fold():
    "Prepares WV French stimuli with vowel labels for 10-fold cross validation."
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'french') * (human_responses['language_stimuli'] == 'FR')]
    human_responses = human_responses.groupby('filename').first().reset_index() # one row per file

    wvFR = process_wv_responses(human_responses)
    wvFR = wvFR.remove_columns('label').rename_column('vowel', 'label')
    prep_wvFR = audio_to_input_values(wvFR)
    prep_wvFR10Fold = k_fold(prep_wvFR, stratify_by_column='label')
    return unannotate_dataset(prep_wvFR10Fold)

def prepare_cc():
    """Consonant Challenge Corpus"""

    cc = datasets.load_dataset('../cc').cast_column('audio', datasets.Audio(sampling_rate=16000)).remove_columns('label')
    consonant_finder = re.compile(r'(?<=\d[aiu])(.*?)(?=[aiu]\d)') # files are named XNVCVN.wav, only three vowels
    cc = cc.map(lambda row: {'label': consonant_finder.search(row['audio']['path']).group()})\
        .map(lambda row: {'label': translators['cc']['wv'][row['label']]})\
        .class_encode_column('label')\
        .shuffle()
    cc = audio_to_input_values(cc)
    cc['dev'] = cc.pop('validation')
    return cc

def prepare_timit_ctc(aligned = False):
    """Prepares TIMIT for CTC sequence classification."""
    timit = datasets.load_dataset('timit_asr', data_dir='../timit/TIMIT')
    timit = timit.remove_columns(['file', 'text', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])
    
    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_remove = ['tcl', 'gcl', 'kcl', 'bcl', 'epi', 'h#', 'dcl', 'q', 'pcl', 'pau']

    def extract_utterances(batch):
        batch['utterance'] = [timit_folding.get(phone, phone) for phone in batch['phonetic_detail']['utterance'] if phone not in timit_remove]
        if aligned:
            batch['start'] = batch['phonetic_detail']['start']
            batch['end'] = batch['phonetic_detail']['stop']
        return batch

    timit = timit.map(extract_utterances, remove_columns=['phonetic_detail'])

    vocab_dict = _get_vocab_dict(timit, 'utterance')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        batch['labels'] = [vocab_dict[phone] for phone in batch['utterance']]
        return batch

    timit = timit.map(prepare_dataset, remove_columns=['audio', 'utterance'])

    return timit, vocab_dict

def prepare_bl_ctc(aligned = False) -> tuple[datasets.DatasetDict, dict[str, int]]:
    """
    Prepares BL-Database and vocab_dict for CTC training.
    
    Returns an aligned dataset if 'aligned' is set to True.
    """
    bl = datasets.load_dataset('../BL-Database/dataset')
    bl = bl['train'].train_test_split() # type: ignore # known to be DatasetDict
    bl = bl.cast_column('audio', datasets.Audio(sampling_rate=16000))

    def extract_utterances(batch):
        batch['phone'] = batch['phonetic_transcription']['phone']
        if aligned:
            batch['start'] = batch['phonetic_transcription']['start']
            batch['end'] = batch['phonetic_transcription']['stop']
        return batch

    bl = bl.map(extract_utterances, remove_columns=['phonetic_transcription'])
    
    vocab_dict = _get_vocab_dict(bl, 'phone')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json', do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        batch['labels'] = processor(text=batch['phone'], is_split_into_words=True, add_special_tokens=False).input_ids
        return batch
    
    bl = bl.map(prepare_dataset, remove_columns=['phone', 'audio'])
    return bl, vocab_dict

def prepare_blEV():
    """Extracts BL syllables into a prepared dataset."""

    bl = cast(DatasetDict, datasets.load_dataset('../BL-Database/dataset'))
    bl = bl['train'] # no existing splits
    with open('human_bl_vowels.json', encoding='utf-8') as f:
        human_bl_vowels: dict[str, str] = json.load(f)
    bl_human_vowels = {v: k for k, v in human_bl_vowels.items()}

    def get_audio_metadata(batch): # not available from annotations; breaks if done during target finding
        return {'length': batch['audio']['array'].size, 'sampling_rate': batch['audio']['sampling_rate']}
    bl_metadata = bl.map(get_audio_metadata, remove_columns = 'audio') # cannot keep audio, raises error
    bl = bl.add_column('length', bl_metadata['length']).add_column('sampling_rate', bl_metadata['sampling_rate'])

    def find_targets(batch):
        utterance: list[str] = batch['phonetic_transcription']['phone']
        # positions and labels of all C+VC+ sequences
        vowel_indices = [i for i, phone in enumerate(utterance) if phone in bl_human_vowels.keys()]
        syllable_start: list[int | None] = [None for i in vowel_indices]
        syllable_end: list[int | None] = [None for i in vowel_indices]
        syllable_vowel = [bl_human_vowels[utterance[i]] for i in vowel_indices]
        for i, vowel_index in enumerate(vowel_indices):
            preceding_vowel_index = next(reversed(vowel_indices[:i]), 0)
            syllable_start[i] = round(batch['phonetic_transcription']['start'][preceding_vowel_index] * batch['sampling_rate'])
            following_vowel_index = next(iter(vowel_indices[i + 1:-1]), None)
            syllable_end[i] = round(batch['phonetic_transcription']['stop'][following_vowel_index] * batch['sampling_rate']) if following_vowel_index else batch['length']
            if syllable_end[i] - syllable_start[i] < batch['sampling_rate'] // 4: # add centred padding to at least a quarter of a second
                syllable_end[i] = min(syllable_end[i] + batch['sampling_rate'] // 8, batch['length']) # do not go past end
                syllable_start[i] = max(0, syllable_end[i] - batch['sampling_rate'] // 4) # do not go before start
                syllable_end[i] = syllable_start[i] + batch['sampling_rate'] // 4 
        batch['syllable_start'] = syllable_start
        batch['syllable_end'] = syllable_end
        batch['syllable_vowel'] = syllable_vowel
        return batch
    bl = bl.map(find_targets)

    def extract_syllables(dataset: Dataset):
        for row in dataset.to_iterable_dataset():
            sampling_rate = row['audio']['sampling_rate']
            for syllable_start, syllable_end, syllable_vowel in zip(row['syllable_start'], row['syllable_end'], row['syllable_vowel']):
                yield {
                    'audio': {
                        'array': row['audio']['array'][syllable_start:syllable_end],
                        'sampling_rate': sampling_rate,
                    },
                    'label': syllable_vowel,
                }
            
    blEV = cast(Dataset, Dataset.from_generator(
        extract_syllables,
        features=datasets.Features({
            'audio': datasets.Audio(sampling_rate=16000, decode=True),
            'label': datasets.Value('string'),
        }),
        gen_kwargs = {'dataset': bl},
    )).class_encode_column('label')
    blEV = blEV.train_test_split(test_size = 0.1, stratify_by_column = 'label', shuffle = True)
    prep_blEV = audio_to_input_values(blEV)
    return unannotate_dataset(prep_blEV)

def prepare_timitEV():
    """Extracts TIMIT syllables into a dataset, which is prepared for training."""
    
    timit = cast(DatasetDict, datasets.load_dataset('timit_asr', data_dir='../timit/TIMIT'))
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        audio_length = batch['phonetic_detail']['stop'][-1]
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in cast(list[str], batch['phonetic_detail']['utterance'])]
        # positions and labels of all C+VC+ sequences (excluding non-target diphthongs)
        target_vowel_indices = [i for i, phone in enumerate(utterance) if phone in target_labels.keys()]
        vowel_indices = [i for i, phone in enumerate(utterance) if phone in timit_vowels]
        syllable_start = [None for i in target_vowel_indices]
        syllable_end = [None for i in target_vowel_indices]
        syllable_vowel = [target_labels[utterance[i]] for i in target_vowel_indices]
        for i, target_index in enumerate(target_vowel_indices):
            preceding_vowel_index = max(target_vowel_indices[:i] + [j for j in vowel_indices if j < target_index] + [0])
            syllable_start[i] = batch['phonetic_detail']['start'][preceding_vowel_index + 1]
            following_vowel_index = min([j for j in target_vowel_indices[i:] if j > target_index] + [j for j in vowel_indices if j > target_index] + [len(utterance)])
            syllable_end[i] = batch['phonetic_detail']['stop'][following_vowel_index - 1]
            syllable_length = syllable_end[i] - syllable_start[i]
            if syllable_length < 4000: # less than an eighth of a second, pad with more context
                syllable_end[i] = min(syllable_end[i] + 2000 - syllable_length//2, audio_length) # centre around centre of target, unless hits edge of sentence
                syllable_start[i] = max(syllable_end[i] - 4000, 0)
                syllable_end[i] = syllable_start[i] + 4000 # in case it was near the start
        batch['syllable_start'] = syllable_start
        batch['syllable_end'] = syllable_end
        batch['syllable_vowel'] = syllable_vowel
        return batch
    timit = timit.map(find_targets) # breaks if we keep original audio feature

    def extract_syllables(dataset: Dataset):
        for row in dataset.to_iterable_dataset():
            sampling_rate = row['audio']['sampling_rate']
            for syllable_start, syllable_end, syllable_vowel in zip(row['syllable_start'], row['syllable_end'], row['syllable_vowel']):
                yield {
                    'audio': {
                        'array': row['audio']['array'][syllable_start:syllable_end],
                        'sampling_rate': sampling_rate,
                    },
                    'label': syllable_vowel
                }
            
    timitEV_test = Dataset.from_generator(
        extract_syllables,
        features = datasets.Features({
            'audio': datasets.Audio(sampling_rate=16000, decode=True),
            'label': datasets.Value('string'),
        }),
        gen_kwargs = {'dataset': timit['test']},
        split = datasets.Split.TEST,
    )
    timitEV_train = Dataset.from_generator(
        extract_syllables,
        features = datasets.Features({
            'audio': datasets.Audio(sampling_rate=16000, decode=True),
            'label': datasets.Value('string'),
        }),
        gen_kwargs = {'dataset': timit['train']},
        split = datasets.Split.TRAIN,
    )
    timitEV = datasets.DatasetDict({'test': timitEV_test, 'train': timitEV_train}).class_encode_column('label')
    prep_timitEV = audio_to_input_values(timitEV)

    return unannotate_dataset(prep_timitEV)

def prepare_timitEC():
    """Extracts TIMIT intervocalic consonant sequences into a dataset, which is prepared for training."""
    
    timit = cast(DatasetDict, datasets.load_dataset('timit_asr', data_dir='../timit/TIMIT'))
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'b': 'b', 'ch': 'tʃ', 'd': 'd', 'dh': 'ð', 'dx': 'd', 'f': 'f', 'g': 'ɡ', 'jh': 'dʒ', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ŋ', 'p': 'p', 'r': 'r', 's': 's', 'sh': 'ʃ', 't': 't', 'th': 'θ', 'v': 'v', 'w': 'w', 'y': 'j', 'z': 'z', 'zh': 'ʒ'} # desired label: timit label

    def find_targets(batch):
        audio_length = batch['phonetic_detail']['stop'][-1]
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in cast(list[str], batch['phonetic_detail']['utterance'])]
        # positions and labels of all V+CV+ sequences (excluding non-target consonants)
        target_consonant_indices = [i for i, phone in enumerate(utterance) if phone in target_labels.keys()]
        consonant_indices = [i for i, phone in enumerate(utterance) if phone in timit_vowels]
        sequence_start = [None for i in target_consonant_indices]
        sequence_end = [None for i in target_consonant_indices]
        sequence_consonant = [target_labels[utterance[i]] for i in target_consonant_indices]
        for i, target_index in enumerate(target_consonant_indices):
            preceding_consonant_index = max(target_consonant_indices[:i] + [j for j in consonant_indices if j < target_index] + [0])
            sequence_start[i] = batch['phonetic_detail']['start'][preceding_consonant_index + 1]
            following_consonant_index = min([j for j in target_consonant_indices[i:] if j > target_index] + [j for j in consonant_indices if j > target_index] + [len(utterance)])
            sequence_end[i] = batch['phonetic_detail']['stop'][following_consonant_index - 1]
            syllable_length = sequence_end[i] - sequence_start[i]
            if syllable_length < 4000: # less than an eighth of a second, pad with more context
                sequence_end[i] = min(sequence_end[i] + 2000 - syllable_length//2, audio_length) # centre around centre of target, unless hits edge of sentence
                sequence_start[i] = max(sequence_end[i] - 4000, 0)
                sequence_end[i] = sequence_start[i] + 4000 # in case it was near the start
        batch['sequence_start'] = sequence_start
        batch['sequence_end'] = sequence_end
        batch['sequence_consonant'] = sequence_consonant
        return batch
    timit = timit.map(find_targets) # breaks if we keep original audio feature

    def extract_syllables(dataset: Dataset):
        for row in dataset.to_iterable_dataset():
            sampling_rate = row['audio']['sampling_rate']
            for sequence_start, sequence_end, sequence_consonant in zip(row['sequence_start'], row['sequence_end'], row['sequence_consonant']):
                yield {
                    'audio': {
                        'array': row['audio']['array'][sequence_start:sequence_end],
                        'sampling_rate': sampling_rate,
                    },
                    'label': sequence_consonant
                }
            
    timitEC_test = Dataset.from_generator(
        extract_syllables,
        features = datasets.Features({
            'audio': datasets.Audio(sampling_rate=16000, decode=True),
            'label': datasets.Value('string'),
        }),
        gen_kwargs = {'dataset': timit['test']},
        split = datasets.Split.TEST,
    )
    timitEC_train = Dataset.from_generator(
        extract_syllables,
        features = datasets.Features({
            'audio': datasets.Audio(sampling_rate=16000, decode=True),
            'label': datasets.Value('string'),
        }),
        gen_kwargs = {'dataset': timit['train']},
        split = datasets.Split.TRAIN,
    )
    timitEC = datasets.DatasetDict({'test': timitEC_test, 'train': timitEC_train}).class_encode_column('label')
    prep_timitEC = audio_to_input_values(timitEC)

    return unannotate_dataset(prep_timitEC)


def prepare_masked_targets():
    """
    Prepares TIMIT utterances, with additional columns 'sample_start' and 'sample_stop'.
    """
    timit = cast(DatasetDict, datasets.load_dataset('timit_asr', data_dir='../timit/TIMIT'))
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in cast(list[str], batch['phonetic_detail']['utterance'])]
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

def prepare_librispeechFR_ctc():
    """
    Prepares Multilingual LibriSpeech French for CTC training. Uses the first 10% of the train set
    """

    train = datasets.load_dataset('facebook/multilingual_librispeech', 'french', split='train[:10%]') # approx. 100 hours
    test = datasets.load_dataset('facebook/multilingual_librispeech', 'french', split='test')
    dev = datasets.load_dataset('facebook/multilingual_librispeech', 'french', split='dev')
    librispeechFR = datasets.DatasetDict({
        'train': train,
        'test': test,
        'dev': dev
    })
    librispeechFR = librispeechFR.remove_columns(
        ['original_path', 'begin_time', 'end_time', 'audio_duration', 'speaker_id', 'chapter_id', 'file', 'id']
    )

    def phonemize(batch): # removes length distinction and lax high front vowel
        text = batch['transcript']
        separator = phonemizer.separator.Separator(word = '', phone = ' ')
        batch['phone'] = [phone.replace('ː', '').replace('ɪ', 'i').split() for phone in phonemizer.phonemize(text, language = 'fr-fr', separator = separator)] # type: ignore # returns a string
        return batch

    librispeechFR = librispeechFR.map(phonemize, batched=True, remove_columns='transcript')
    
    def remove_language_switching(batch): # remove any utterances that contain non-french phonemizations
        to_keep = [i for i, phone in enumerate(batch['phone']) if '(fr)' not in phone]
        return {'to_keep': [to_keep]}
    
    to_keep = librispeechFR.map(remove_language_switching, batched=True, batch_size=-1, remove_columns=librispeechFR['train'].column_names)
    for split in ['train', 'test', 'dev']:
        librispeechFR[split] = librispeechFR[split].select(to_keep[split]['to_keep'][0])
    
    vocab_dict = _get_vocab_dict(librispeechFR, 'phone')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json', do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        with processor.as_target_processor():
            batch['labels'] = processor(batch['phone'], is_split_into_words=True, add_special_tokens=False).input_ids
        return batch

    librispeechFR = librispeechFR.map(prepare_dataset, batched=True, batch_size=250, remove_columns=librispeechFR['train'].column_names)

    return librispeechFR, vocab_dict

def prepare_librispeech_ctc(classic=False, word_boundaries=False):
    """
    Prepares Multilingual LibriSpeech English 10k for CTC training. Uses the first 1% of the train set (approx. 100 hours).
    
    If classic = True, prepares classic LibriSpeech 100h instead.
    """

    if classic:
        librispeech = datasets.load_dataset('openslr/librispeech_asr', verification_mode='no_checks', data_dir='all', data_files={
            'train.clean.100': 'train.clean.100/*.parquet',
            'validation.clean': 'validation.clean/*.parquet'
        })
        librispeech = librispeech.remove_columns(
            ['file', 'id', 'speaker_id', 'chapter_id']
        )
    else:
        librispeech = datasets.load_dataset('parler-tts/mls_eng_10k', verification_mode='no_checks', data_dir='data', data_files={
            'train': [f'train-0000{i}-of-00317.parquet' for i in range(3)],
            'test': 'test-00000-of-00001.parquet',
            'dev': 'dev-00000-of-00001.parquet'
        })
        librispeech = librispeech.remove_columns(
            ['original_path', 'begin_time', 'end_time', 'audio_duration', 'speaker_id', 'book_id']
        )
    
    librispeech = cast(DatasetDict, librispeech)

    folding = {'ə': 'ʌ', 'ɐ': 'ʌ', 'ᵻ': 'ɪ', 'əl': 'l', 'ɚ': 'ɹ', 'n̩': 'n', 'ææ': 'æ', 'ɑ̃': 'ɑ', 'o': 'oʊ', 'x': 'k', 'r': 'ɹ'}

    transcript_column = 'text' if classic else 'transcript' # column name changes
    def phonemize(batch): # removes length marker, ensures rhotic vowels are separate phonemes
        text = batch[transcript_column]
        separator = phonemizer.separator.Separator(word = '|' if word_boundaries else '', phone = ' ')
        batch['phone'] = [[folding.get(phone, phone) for phone in phones.replace('ː', '').replace('ɹ', ' ɹ').replace('ɚ', ' ɚ').replace('ɬ', 'ʃ l').replace('ɔ̃', 'ɔ n').replace('aɪə', 'aɪ ə').replace('ɡʲ', 'ɡ j').replace('ʔ', 'ɾ').replace('ɜ ɹ', 'ɚ').replace('ɜ', 'ɚ').replace('iə', 'j ə').split()] for phones in phonemizer.phonemize(text, separator = separator)] # type: ignore # returns a string
        return batch

    librispeech = librispeech.map(phonemize, batched=True, remove_columns=transcript_column)

    def extract_phones(batch):
        all_phones = sum(batch['phone'], [])
        vocab = list(set(all_phones))
        return {'vocab': [vocab], 'all_phones': [all_phones]}
    
    vocab_dict = _get_vocab_dict(librispeech, 'phone')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json', do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = [audio['array'] for audio in batch['audio']]
        batch['input_values'] = processor(audio, sampling_rate=batch['audio'][0]['sampling_rate']).input_values
        batch['labels'] = processor(text=batch['phone'], is_split_into_words=True, add_special_tokens=False).input_ids
        return batch

    librispeech = librispeech.map(prepare_dataset, batched=True, batch_size=250, remove_columns=librispeech[list(librispeech.keys())[0]].column_names)

    return librispeech, vocab_dict

def _get_vocab_dict(dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict, text_column: str) -> dict[str, int]:
    """
    Given a dataset with splits and transcriptions, returns a vocab_dict.
    
    Args:
        dataset: A dataset with splits, with at least a column containing
            the text to be converted to a vocabulary.
        text_column: The name of the column containing the text. This can
            be a column of pretokenized lists of strings, or just a
            column of strings. In the latter case, each character
            is treated as a separate token.
    """
    
    vocab_set = set()
    
    def _extract_tokens(batch):
        vocab_set.update(*batch[text_column])
        return None
    
    dataset.map(_extract_tokens, batched = True)
    vocab_list = list(vocab_set)
    vocab_list.extend(['|', '<unk>', '<pad>']) # special tokens
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    return vocab_dict

@overload
def audio_to_input_values(dataset: Dataset, feature_extractor = None) -> Dataset: ...
@overload
def audio_to_input_values(dataset: DatasetDict, feature_extractor = None) -> DatasetDict: ...
def audio_to_input_values(dataset: Dataset | DatasetDict, feature_extractor = None) -> Dataset | DatasetDict:
    """Removes 'audio' column and adds an 'input_values' column using feature_extractor."""

    if not feature_extractor:
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    
    def _generate_input_values(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    dataset = dataset.map(_generate_input_values, remove_columns = 'audio')
    return dataset
