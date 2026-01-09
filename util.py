from typing import Mapping, Sequence
from spec import TestDatasetSpec, ProbabilitySpec, HumanProbabilitySpec, _SEPARATOR
from probabilities_handler import Probabilities, HumanProbabilities, TestDataset, _PROBABILITIES_PATH, _PROBABILITIES_PREFIX
import glob
import warnings

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

class ProbabilitiesMap(Mapping):
    """
    Mapping for probabilities.
    
    If autoload, prints the autoloaded probabilities.
    """
    
    def __init__(self, map = (), autoload = False):
        self._map = dict(map)
        
        if autoload:
            self.autoload_probabilities()

    def load(self, spec: str) -> None:
        try:
            self._map[spec] = Probabilities(spec)
        except (ValueError, NotImplementedError):
            try:
                self._map[spec] = HumanProbabilities(spec)
            except (ValueError, NotImplementedError):
                raise ValueError(f"Invalid probability spec: {spec}.")

    def __getitem__(self, key: str | ProbabilitySpec | HumanProbabilitySpec | list[str]):
        if not isinstance(key, str) or isinstance(key, list) and all(isinstance(k, str) for k in key):
            raise TypeError('Probability specification must be a string or list of strings!')
        
        if isinstance(key, list):
            return [self.__getitem__(k) for k in key]
        
        if key not in self._map.keys():
            return self.__missing__(key)
        
        return self._map[key]

    def __missing__(self, key: str):
        try:
            self.load(key)
            return self._map[key]
        except ValueError:
            raise NotImplementedError(f'Could not find or generate probabilities {key}.')
    
    def __iter__(self):
        return iter(self._map)
    
    def __len__(self) -> int:
        return len(self._map)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self._map)})'
    
    def autoload_probabilities(self, path: str = _PROBABILITIES_PATH) -> None:
        paths = glob.glob(f'{_PROBABILITIES_PREFIX}{_SEPARATOR}*.csv', root_dir=path)
        for path in paths:
            try:
                self.load(path[2:-4])
            except ValueError:
                warnings.warn(f"Failed to load probability with spec {path[2:-4]}. Continuing...", UserWarning)

