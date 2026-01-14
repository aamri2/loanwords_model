from spec import TestDatasetSpec
import pandas as pd
from typing import Mapping, Sequence

_WV_DATA_PATH = 'WorldVowels_stimuli.csv'
_WV_FORMANTS_PATH = '../stimuli_world_vowels/formants/world_vowels_formants.csv'

def load_wv_data(path = _WV_DATA_PATH) -> pd.DataFrame:
    """Loads and prepares WV data for use as a test dataset."""

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

    contexts = dataset.dataset.loc[:, ['file', 'prev_phone', 'next_phone', 'context']]
    contexts = contexts.set_index('file')
    return contexts

class TestDataset():
    """Uses TestDatasetSpec to load and prepare test datasets."""

    spec: TestDatasetSpec
    dataset: pd.DataFrame
    _test_dataset_loader = {'wv': load_wv_data}
    formants: pd.DataFrame
    _formant_loader = {'wv': load_wv_formants}
    contexts: pd.DataFrame
    _context_loader = {'wv': load_wv_contexts}

    def __init__(self, spec: str | TestDatasetSpec):
        self.spec = TestDatasetSpec(spec)
        self.dataset = self.load_dataset()
        self.formants = self.load_formants()
        self.contexts = self.load_contexts()

    def load_dataset(self):
        try:
            return self._test_dataset_loader[self.spec.value]()
        except KeyError:
            raise NotImplementedError(f"Cannot load test dataset {self.spec}.")
    
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

    def __getitem__(self, key: str | TestDatasetSpec | Sequence[str | TestDatasetSpec]):
        if not (isinstance(key, (str, TestDatasetSpec)) or (isinstance(key, Sequence) and all(isinstance(k, (str, TestDatasetSpec)) for k in key))):
            raise TypeError('Probability specification must be a string or list of strings!')
        
        if isinstance(key, Sequence):
            return [self.__getitem__(k) for k in key]
        
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