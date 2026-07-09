from spec import TestDatasetSpec
from model_handler import Model
from dataset_handler import audio_to_input_values, translators
from base_model_handler import Base
import pandas as pd
from typing import Mapping, Sequence, overload
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio
from transformers import Wav2Vec2FeatureExtractor
from functools import cache
import glob

_WV_AUDIO_PATH = '../stimuli_world_vowels/'
_WV_METADATA_PATH = '../human_vowel_responses.csv'
_WV_FORMANTS_PATH = '../stimuli_world_vowels/formants/world_vowels_formants.csv'

_CC_AUDIO_PATH = '../stimuli_consonant_challenge/'

def load_wv_dataset(path = _WV_AUDIO_PATH) -> Dataset:
    """Loads and prepares WV Dataset for use as a test dataset (without labels)."""
    
    metadata = pd.read_csv(_WV_METADATA_PATH).groupby('filename').first()
    dataset = Dataset.from_dict({
        'audio': path + metadata.index + '.wav',
        'language': metadata['language_stimuli'],
        'vowel': metadata['#phone'],
        'file': metadata.index,
        'prev_phone': metadata['prev_phone'],
        'next_phone': metadata['next_phone'],
        'context': metadata['prev_phone'] + '_' + metadata['next_phone'],
    }).cast_column('audio', Audio())

    formants = pd.read_csv(_WV_FORMANTS_PATH)
    formants['F1_norm'] = formants['F1'] / formants['F3']
    formants['F2_norm'] = formants['F2'] / formants['F3']
    formants = formants.set_index('file')
    for name in formants.columns:
        dataset = dataset.add_column(name=name, column=formants[name].reindex(dataset['file']))
    dataset = audio_to_input_values(dataset, Wav2Vec2FeatureExtractor())
    return dataset

def load_cc_dataset(path = _CC_AUDIO_PATH) -> Dataset:
    files = pd.Series([f.removesuffix('.wav') for f in glob.glob('*.wav', root_dir=path)])
    dataset = Dataset.from_dict({
        'audio': path + files + '.wav',
        'prev_phone': files.str.extract(r'(?<=\d)([aiu])', expand=False),
        'next_phone': files.str.extract(r'([aiu])(?=\d)', expand=False),
        'consonant': files.str.extract(r'(?<=[aiu])(.*?)(?=[aiu])', expand=False).map(translators['cc']['wv']),
        'file': files,
    }).cast_column('audio', Audio(sampling_rate=16000))
    dataset = audio_to_input_values(dataset, Wav2Vec2FeatureExtractor())
    return dataset

class TestDataset():
    """Uses TestDatasetSpec to load and prepare test datasets."""

    spec: TestDatasetSpec
    dataset: Dataset
    _dataset_loader = {'wv': load_wv_dataset, 'cc': load_cc_dataset}

    def __init__(self, spec: str | TestDatasetSpec):
        self.spec = TestDatasetSpec(spec)
        self.dataset = self.load_dataset()

    def load_dataset(self) -> Dataset:
        try:
            return self._dataset_loader[self.spec.value]()
        except KeyError:
            raise NotImplementedError(f"Cannot load test dataset {self.spec}.")

class TestDatasetMap(Mapping):
    """Mapping for test datasets."""

    def __init__(self, map=()):
        self._map = dict(map)

    @overload
    def __getitem__(self, key: str | TestDatasetSpec) -> TestDataset: ...
    @overload
    def __getitem__(self, key: Sequence[str | TestDatasetSpec]) -> tuple[TestDataset, ...]: ...
    def __getitem__(self, key: str | TestDatasetSpec | Sequence[str | TestDatasetSpec]) -> TestDataset | tuple[TestDataset, ...]:
        if not (isinstance(key, (str, TestDatasetSpec)) or (isinstance(key, Sequence) and all(isinstance(k, (str, TestDatasetSpec)) for k in key))):
            raise TypeError('Probability specification must be a string or list of strings!')
        
        if isinstance(key, Sequence) and not isinstance(key, str):
            return tuple(self.__getitem__(k) for k in key)
        
        key = str(key)
        if key not in self._map.keys():
            return self.__missing__(key)
        
        return self._map[key]

    def __missing__(self, key: str):
        self._map[key] = TestDataset(key)
        return self._map[key]
        
    def __iter__(self):
        return iter(self._map)
    
    def __len__(self):
        return len(self._map)

t = TestDatasetMap()