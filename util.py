from probabilities_handler import Probabilities, HumanProbabilities, _PROBABILITIES_PATH, _PROBABILITIES_PREFIX
from typing import Mapping, Sequence, overload, cast, Any, Iterable
from spec import ProbabilitySpec, HumanProbabilitySpec, _SEPARATOR
from model_handler import _MODEL_PATH, _MODEL_PREFIX
import glob
import warnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

AnyProbabilitiesSpec = str | ProbabilitySpec | HumanProbabilitySpec

class ProbabilitiesMap(Mapping):
    """
    Mapping for probabilities.
    
    If autoload, prints the autoloaded probabilities.
    """
    
    def __init__(self, map = (), autoload = False, path: str = _PROBABILITIES_PATH):
        self._map: dict[AnyProbabilitiesSpec, Probabilities | HumanProbabilities] = dict(map)
        self.path = path
        
        if autoload:
            self.autoload_probabilities()

    def load(self, spec: AnyProbabilitiesSpec, model_kwargs: dict[str, Any] = {}) -> None:
        spec = str(spec)
        try:
            self._map[spec] = Probabilities(spec, path=self.path, model_kwargs=model_kwargs)
        except (ValueError, NotImplementedError):
            try:
                self._map[spec] = HumanProbabilities(spec, path=self.path)
            except (ValueError, NotImplementedError):
                raise ValueError(f"Invalid probability spec: {spec}.")

    @overload
    def __getitem__(self, key: str | ProbabilitySpec) -> Probabilities: ...
    @overload
    def __getitem__(self, key: str | HumanProbabilitySpec) -> HumanProbabilities: ...
    @overload
    def __getitem__(self, key: Iterable[AnyProbabilitiesSpec]) -> list[Probabilities | HumanProbabilities]: ...
    def __getitem__(self, key: AnyProbabilitiesSpec | Iterable[AnyProbabilitiesSpec]) -> Probabilities | HumanProbabilities | list[Probabilities | HumanProbabilities]:
        if not isinstance(key, (str, ProbabilitySpec, HumanProbabilitySpec)):
            if not isinstance(key, Iterable) or any(not isinstance(k, (AnyProbabilitiesSpec)) for k in key):
                raise TypeError('Probability specification must be a string or list of strings!')
            elif isinstance(key, Iterable):
                return [self.__getitem__(k) for k in key]
        
        if key not in self._map.keys():
            return self.__missing__(key)
        
        return self._map[key]

    def __missing__(self, key: AnyProbabilitiesSpec):
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
    
    def autoload_probabilities(self) -> None:
        probabilities = self.find_probabilities()
        for probability in probabilities:
            try:
                self.load(probability)
            except ValueError:
                warnings.warn(f"Failed to load probability with spec {probability[2:-4]}. Continuing...", UserWarning)

    def find_probabilities(self) -> tuple[str, ...]:
        paths = glob.glob(f'{_PROBABILITIES_PREFIX}{_SEPARATOR}*.csv', root_dir=self.path)
        probabilities = tuple(path[2: - 4] for path in paths)
        return probabilities
    
    def heatmap(self, keys: Sequence[AnyProbabilitiesSpec], *args: str, **kwargs: str | Iterable[str]):
        df = pd.concat((self[key].pool(*args, **kwargs) for key in keys), keys=[str(key) for key in keys]).reset_index(level=0,names='model')
        g = sns.FacetGrid(df, col_wrap=min(len(keys), 3), col='model', height=2.5)
        g.map_dataframe(lambda data, **kwargs: sns.heatmap(data[[column for column in df.columns if column != 'model']], **kwargs), cmap = 'crest', square = True, xticklabels = True, yticklabels = True)
        plt.suptitle(f'where {kwargs}', wrap=True)
        plt.subplots_adjust(top=.9)
        plt.show(block = False)
    
    @overload
    def js_divergence(self, key1: AnyProbabilitiesSpec, key2: AnyProbabilitiesSpec, *args, **kwargs) -> np.ndarray: ...
    @overload
    def js_divergence(self, key1: AnyProbabilitiesSpec, key2: Iterable[AnyProbabilitiesSpec], *args, **kwargs) -> list[np.ndarray]: ...
    def js_divergence(self, key1: AnyProbabilitiesSpec, key2: AnyProbabilitiesSpec | Iterable[AnyProbabilitiesSpec], *args: str, **kwargs: str | Iterable[str]) -> np.ndarray | list[np.ndarray]:
        """The Jensen-Shannon Divergences between two pooled probabilities, by row."""

        if isinstance(key2, Iterable) and not isinstance(key2, AnyProbabilitiesSpec):
            return [self.js_divergence(key1, key, *args, **kwargs) for key in key2]
        p = self[key1].pool(*args, **kwargs)
        q = self[key2].pool(*args, **kwargs)
        common_index = p.index.intersection(q.index)

        js_divergences = jensenshannon(p.loc[common_index], q.loc[common_index], axis=1) ** 2 # square JS distance to get JS divergence
        return js_divergences
    
    @overload
    def accuracy(self, key1: AnyProbabilitiesSpec, key2: AnyProbabilitiesSpec, *args, **kwargs) -> float: ...
    @overload
    def accuracy(self, key1: AnyProbabilitiesSpec, key2: Iterable[AnyProbabilitiesSpec], *args, **kwargs) -> list[float]: ...
    def accuracy(self, key1: AnyProbabilitiesSpec, key2: AnyProbabilitiesSpec | Iterable[AnyProbabilitiesSpec], *args, **kwargs) -> float | list[float]:
        """The percentage of shared row-wise argmaxes between the first and second probabilities."""

        if isinstance(key2, Iterable) and not isinstance(key2, AnyProbabilitiesSpec):
            return [self.accuracy(key1, key, *args, **kwargs) for key in key2]
        p = self[key1].pool(*args, **kwargs)
        q = self[key2].pool(*args, **kwargs)
        common_index = p.index.intersection(q.index)

        p_classifications = p.loc[common_index].apply(lambda x: x.argmax(), axis=1)
        q_classifications = q.loc[common_index].apply(lambda x: x.argmax(), axis=1)
        accuracy = (p_classifications == q_classifications).astype(int).mean()
        return accuracy
    
    def entropy_histogram(self, key: Iterable[AnyProbabilitiesSpec], *args, **kwargs):
        for probability in key:
            self[probability].entropy_histogram(*args, **kwargs)
    
    def js_divergence_histogram(self, key1: AnyProbabilitiesSpec, key2: AnyProbabilitiesSpec | Iterable[AnyProbabilitiesSpec], *args, **kwargs):
        if not isinstance(key2, str) and isinstance(key2, Sequence):
            df = pd.DataFrame({key: self.js_divergence(key1, key, *args, **kwargs) for key in key2}).melt(var_name='p', value_name='JS-div')
            g = sns.displot(df, x='JS-div', col='p', col_wrap=min(len(key2), 3))
            def mean_lines(x, **kwargs):
                plt.axvline(x.mean(), linestyle = 'dashed', linewidth = 1)
                plt.text(x.mean()*1.1, plt.ylim()[1]*0.9, f'Mean: {x.mean():.2f}')
            g.map(mean_lines, 'JS-div')
            plt.subplots_adjust(top=.9)
            plt.suptitle(f'JS-divs with {key1}')
            plt.show(block=False)
            return
        plt.figure()
        js_divergence = self.js_divergence(key1, key2, *args, **kwargs)
        sns.histplot(js_divergence)
        mean_js_divergence = sum(js_divergence)/len(js_divergence)
        plt.axvline(x = mean_js_divergence, linestyle = 'dashed', linewidth = 1)
        plt.text(mean_js_divergence*1.1, plt.ylim()[1]*0.9, f'Mean: {mean_js_divergence:.2f}')
        plt.title(f'JS-divs btwn {key1}, {key2}', wrap=True)
        plt.show(block=False)

def run_new_models(p: ProbabilitiesMap, model_path=_MODEL_PATH):
    """Runs and saves all classification (contains class) models in a given path on wv."""

    models = [model[len(_MODEL_PREFIX + _SEPARATOR):] for model in glob.glob(f'{_MODEL_PREFIX}*class*', root_dir=model_path)] # remove m_
    old_models = [probability[:-3] for probability in p.find_probabilities()] # remove _wv
    new_models = set(models).difference(old_models)
    for model in tqdm(new_models, desc='Running models'):
        p.load(f'{model}_wv', model_kwargs={'path': model_path})
        p[f'{model}_wv']\
            .probabilities[['probabilities', 'classification', 'language', 'vowel', 'file']]\
            .to_csv(f'{p.path}{_PROBABILITIES_PREFIX}{_SEPARATOR}{model}_wv.csv')

def mean_cross_validations(p: ProbabilitiesMap):
    """
    For cross-validated probabilities where the mean isn't present, calculate and save the mean.
    Assumes always 10 folds from 0 to 9.
    """

    cross_probabilities = set(probability[:-5] for probability in p.find_probabilities() if 'cross' in probability) # remove _i_wv
    nomean_cross_probabilities = [probability for probability in cross_probabilities if f'{probability}_N_wv' not in p.find_probabilities()]
    for probability in tqdm(nomean_cross_probabilities, desc='Averaging probabilities'):
        ps = [p[f'{probability}_{i}_wv'].probabilities for i in range(10)]
        p0 = ps[0]
        p0['probabilities'] = pd.concat([pi['probabilities'] for pi in ps], axis = 1).mean(axis = 1)
        p0[['probabilities', 'classification', 'language', 'vowel', 'file']].to_csv(f'{p.path}{_PROBABILITIES_PREFIX}{_SEPARATOR}{probability}_N_wv.csv')