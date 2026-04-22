from typing import Mapping, Sequence, overload, cast
from spec import ProbabilitySpec, HumanProbabilitySpec, _SEPARATOR
from probabilities_handler import Probabilities, HumanProbabilities, _PROBABILITIES_PATH, _PROBABILITIES_PREFIX
from model_handler import _MODEL_PATH, _MODEL_PREFIX
import glob
import warnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import pandas as pd
from tqdm import tqdm

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
        probabilities = self.find_probabilities(path = path)
        for probability in probabilities:
            try:
                self.load(probability)
            except ValueError:
                warnings.warn(f"Failed to load probability with spec {probability[2:-4]}. Continuing...", UserWarning)

    def find_probabilities(self, path: str = _PROBABILITIES_PATH) -> tuple[str, ...]:
        paths = glob.glob(f'{_PROBABILITIES_PREFIX}{_SEPARATOR}*.csv', root_dir=path)
        probabilities = tuple(path[2: - 4] for path in paths)
        return probabilities
    
    def heatmap(self, keys: Sequence[str | ProbabilitySpec | HumanProbabilitySpec], *args, **kwargs):
        df = pd.concat((self[key].pool(*args, **kwargs) for key in keys), keys=[str(key) for key in keys]).reset_index(level=0,names='p')
        g = sns.FacetGrid(df, col_wrap=min(len(keys), 3), col='p', height=2.5)
        g.map_dataframe(lambda data, **kwargs: sns.heatmap(data[[column for column in df.columns if column != 'p']], **kwargs), cmap = 'crest', square = True, xticklabels = True, yticklabels = True)
        plt.suptitle(f'where {kwargs}', wrap=True)
        plt.subplots_adjust(top=.9)
        plt.show(block = False)
    
    def js_divergence(self, key1: str | ProbabilitySpec | HumanProbabilitySpec, key2: str | ProbabilitySpec | HumanProbabilitySpec, *args, **kwargs) -> list[float]:
        """The Jensen-Shannon Divergences between two pooled probabilities, by row."""

        p = self[key1].pool(*args, **kwargs)
        q = self[key2].pool(*args, **kwargs)
        js_divergences = [jensenshannon(p.iloc[row], q.iloc[row])**2 for row in range(len(p))] # square JS distance to get JS divergence
        return cast(list[float], js_divergences)
    
    def entropy_histogram(self, key: Sequence[str | ProbabilitySpec | HumanProbabilitySpec], *args, **kwargs):
        for probability in key:
            self[probability].entropy_histogram(*args, **kwargs)
    
    def js_divergence_histogram(self, key1: str | ProbabilitySpec | HumanProbabilitySpec, key2: str | ProbabilitySpec | HumanProbabilitySpec | Sequence[str | ProbabilitySpec | HumanProbabilitySpec], *args, **kwargs):
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

def run_new_models(p: ProbabilitiesMap):
    """Runs and saves all classification (contains class) models in _MODEL_DIR on wv."""

    models = [model[len(_MODEL_PREFIX + _SEPARATOR):] for model in glob.glob(f'{_MODEL_PREFIX}*class*', root_dir=_MODEL_PATH)] # remove m_
    old_models = [probability[:-3] for probability in p.find_probabilities()] # remove _wv
    new_models = set(models).difference(old_models)
    for model in tqdm(new_models, desc='Running models'):
        p[f'{model}_wv']\
            .probabilities[['probabilities', 'classification', 'language', 'vowel', 'file']]\
            .to_csv(f'{_PROBABILITIES_PATH}{_PROBABILITIES_PREFIX}{_SEPARATOR}{model}_wv.csv')

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
        p0[['probabilities', 'classification', 'language', 'vowel', 'file']].to_csv(f'{_PROBABILITIES_PATH}{_PROBABILITIES_PREFIX}{_SEPARATOR}{probability}_N_wv.csv')