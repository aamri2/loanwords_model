from typing import Mapping, Sequence, overload
from spec import TestDatasetSpec, ProbabilitySpec, HumanProbabilitySpec, _SEPARATOR
from probabilities_handler import Probabilities, HumanProbabilities, _PROBABILITIES_PATH, _PROBABILITIES_PREFIX
import glob
import warnings

class ProbabilitiesMap(Mapping):
    """
    Mapping for probabilities.
    
    If autoload, prints the autoloaded probabilities.
    """
    
    def __init__(self, map = (), autoload = False):
        self._map: dict[str | ProbabilitySpec | HumanProbabilitySpec, Probabilities] = dict(map)
        
        if autoload:
            self.autoload_probabilities()

    def load(self, spec: str | ProbabilitySpec | HumanProbabilitySpec) -> None:
        spec = str(spec)
        try:
            self._map[spec] = Probabilities(spec)
        except (ValueError, NotImplementedError):
            try:
                self._map[spec] = HumanProbabilities(spec)
            except (ValueError, NotImplementedError):
                raise ValueError(f"Invalid probability spec: {spec}.")

    @overload
    def __getitem__(self, key: str | ProbabilitySpec) -> Probabilities: ...
    @overload
    def __getitem__(self, key: str | HumanProbabilitySpec) -> HumanProbabilities: ...
    @overload
    def __getitem__(self, key: Sequence[str | ProbabilitySpec | HumanProbabilitySpec]) -> list[Probabilities | HumanProbabilities]: ...
    def __getitem__(self, key: str | ProbabilitySpec | HumanProbabilitySpec | Sequence[str | ProbabilitySpec | HumanProbabilitySpec]) -> Probabilities | HumanProbabilities | list[Probabilities | HumanProbabilities]:
        if not isinstance(key, (str, ProbabilitySpec, HumanProbabilitySpec)):
            if not isinstance(key, Sequence) or any(not isinstance(k, (str, ProbabilitySpec, HumanProbabilitySpec)) for k in key):
                raise TypeError('Probability specification must be a string or list of strings!')
            elif isinstance(key, Sequence):
                return [self.__getitem__(k) for k in key]
        
        if key not in self._map.keys():
            return self.__missing__(key)
        
        return self._map[key]

    def __missing__(self, key: str | ProbabilitySpec | HumanProbabilitySpec):
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

