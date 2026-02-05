from spec import TestDatasetSpec
from model_handler import Model
from dataset_handler import audio_to_input_values
from base_model_handler import Base
import pandas as pd
from typing import Mapping, Sequence, overload
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio
from transformers import Wav2Vec2FeatureExtractor
from functools import cache

_WV_AUDIO_PATH = '../stimuli_world_vowels/'
_WV_METADATA_PATH = 'WorldVowels_stimuli.csv'
_WV_FORMANTS_PATH = '../stimuli_world_vowels/formants/world_vowels_formants.csv'

def load_wv_dataset(path = _WV_AUDIO_PATH) -> Dataset:
    """Loads and prepares WV Dataset for use as a test dataset (without labels)."""
    
    metadata = pd.read_csv(_WV_METADATA_PATH)
    with open(path + 'audio_files.txt') as f:
        files = f.readlines()
    files = [file.strip() for file in files]
    metadata = metadata[metadata['#file_extract'].isin(files)]
    dataset = Dataset.from_dict({
        'audio': [f'{path}{file}' for file in metadata['#file_extract'] if file in files],
        'language': metadata['language'],
        'vowel': metadata['#phone'],
        'file': metadata['index']
    }).cast_column('audio', Audio())

    dataset = audio_to_input_values(dataset, Wav2Vec2FeatureExtractor())

    return dataset

def load_wv_metadata(path = _WV_METADATA_PATH) -> pd.DataFrame:
    """Loads and prepares WV metadata for use as a test dataset."""

    return pd.read_csv(path).rename({'index': 'file'}, axis = 1)

def load_wv_formants(dataset: 'TestDataset') -> pd.DataFrame:
    """Loads and prepares WV formants in a DataFrame."""

    formants = pd.read_csv(_WV_FORMANTS_PATH)
    formants['F1_norm'] = formants['F1'] / formants['F3']
    formants['F2_norm'] = formants['F2'] / formants['F3']
    formants = formants.set_index('file')
    return formants

def load_wv_contexts(dataset: 'TestDataset') -> pd.DataFrame:
    """Loads and prepares WV contexts in a DataFrame."""

    contexts = dataset.metadata.loc[:, ['file', 'prev_phone', 'next_phone', 'context']]
    contexts = contexts.set_index('file')
    return contexts

class TestDataset():
    """Uses TestDatasetSpec to load and prepare test datasets."""

    spec: TestDatasetSpec
    dataset: Dataset
    metadata: pd.DataFrame
    formants: pd.DataFrame
    contexts: pd.DataFrame
    _dataset_loader = {'wv': load_wv_dataset}
    _metadata_loader = {'wv': load_wv_metadata}
    _formant_loader = {'wv': load_wv_formants}
    _context_loader = {'wv': load_wv_contexts}

    def __init__(self, spec: str | TestDatasetSpec):
        self.spec = TestDatasetSpec(spec)
        self.dataset = self.load_dataset()
        self.metadata = self.load_metadata()
        self.formants = self.load_formants()
        self.contexts = self.load_contexts()

    def load_dataset(self) -> Dataset:
        try:
            return self._dataset_loader[self.spec.value]()
        except KeyError:
            raise NotImplementedError(f"Cannot load test dataset {self.spec}.")

    def load_metadata(self) -> pd.DataFrame:
        try:
            return self._metadata_loader[self.spec.value]()
        except KeyError:
            raise NotImplementedError(f"Cannot load metadata for test dataset {self.spec}.")
    
    def load_formants(self) -> pd.DataFrame:
        try:
            return self._formant_loader[self.spec.value](self)
        except KeyError:
            raise NotImplementedError(f"Cannot load formants for test dataset {self.spec}.")
    
    def load_contexts(self) -> pd.DataFrame:
        try:
            return self._context_loader[self.spec.value](self)
        except KeyError:
            raise NotImplementedError(f"Cannot load contexts for test dataset {self.spec}.")

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
        try:
            self._map[key] = TestDataset(key)
            return self._map[key]
        except FileNotFoundError:
            raise NotImplementedError(f'Cannot dynamically create test dataset {key}.')
    
    def __iter__(self):
        return iter(self._map)
    
    def __len__(self):
        return len(self._map)

t = TestDatasetMap()