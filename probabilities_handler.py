from spec import ProbabilitySpec, HumanProbabilitySpec, _SEPARATOR
from datasets import Dataset
from test_dataset_handler import TestDataset
from model_handler import Model
from model_handler import probabilities as legacy_probabilities
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import cast

_PROBABILITIES_PATH = 'probabilities/'
_PROBABILITIES_PREFIX = 'p'

class Probabilities():
    """
    Uses ProbabilitySpec to load and parse model probabilities, and
    process and display them in various ways.
    """

    spec: ProbabilitySpec
    test_dataset: TestDataset
    probabilities: pd.DataFrame

    def __init__(self, spec: str | ProbabilitySpec):
        self.spec = ProbabilitySpec(spec)
        self.test_dataset = TestDataset(self.spec.test_dataset)
        self.probabilities = self.load_probabilities()

    def pool(self, *args: str, **kwargs: str) -> pd.DataFrame:
        """
        Returns a pivot table in wide format.
        
        Args:
            *args: Columns to pool across.
            **kwargs: Columns to filter by.
        """
        probabilities = self.probabilities

        if kwargs:
            filter = zip(*[probabilities[column] == value for column, value in kwargs.items()])
            probabilities = probabilities[[all(row) for row in filter]]
        
        pooled_probabilities = probabilities.pivot_table(values = 'probabilities', columns = 'classification', index = args, sort = False)
        return pooled_probabilities
    
    def heatmap(self, *args: str, **kwargs: str):
        """Uses pool to create a seaborn heatmap."""

        sns.heatmap(self.pool(*args, **kwargs), cmap = 'crest', square = True)
        plt.show()

    def load_probabilities(self, path = _PROBABILITIES_PATH, prefix = _PROBABILITIES_PREFIX) -> pd.DataFrame:
        """Loads fully-prepared probabilities from a specification. Attempts to create them if missing."""

        try:
            probabilities = pd.read_csv(f'{path}{prefix}{_SEPARATOR}{self.spec}.csv')
        except:
            if str(self.spec) == 'w2v2_ctc_1_timit_max_class_3_wvResponses_wv':
                dataset = self.test_dataset.dataset
                probabilities = legacy_probabilities(Model(self.spec.model).model, self.test_dataset.dataset)
            else:
                raise NotImplementedError("Cannot dynamically generate probabilities yet.")
        
        probabilities = self.prepare_probabilities(probabilities)
        return probabilities


    def prepare_probabilities(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Prepares probabilities from the file."""

        probabilities = self._add_formants(probabilities)
        probabilities = self._add_contexts(probabilities)
        probabilities = world_vowel_sort(probabilities)
        return probabilities
    
    def _add_formants(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """
        Returns probabilities with formant columns added.
        
        Adds columns F1, F2, F3, F1_norm, and F2_norm.
        """

        formants = self.test_dataset.formants
        formants['F1_norm'] = formants['F1'] / formants['F3']
        formants['F2_norm'] = formants['F2'] / formants['F3']

        probabilities = probabilities.join(formants, on = 'file', validate = 'many_to_one')
        return probabilities
    
    def _add_contexts(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Returns the probabilities with added columns prev_phone, next_phone"""

        contexts = self.test_dataset.contexts
        probabilities = probabilities.join(contexts, on = 'file', validate = 'many_to_one')
        return probabilities
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(str(self.spec))})'

vowel_order = 'iɪyʏeɛøœaæɐɑʌoɔɤuʊɯ:ː\u0303'
def world_vowel_sort(data: pd.DataFrame):
    data = data.sort_values(
        by = 'classification',
        key = lambda x: [[vowel_order.index(c) for c in s] for s in x],
    ).sort_values(by = 'file', kind = 'mergesort').sort_values(
        by = 'vowel',
        key = lambda x: [[vowel_order.index(c) for c in s] for s in x],
        kind = 'mergesort'
    ).sort_values(by = ['language'], kind = 'mergesort')
    return data

class HumanProbabilities(Probabilities):
    spec: HumanProbabilitySpec

    def __init__(self, spec: str | HumanProbabilitySpec):
        self.spec = HumanProbabilitySpec(spec)
        self.test_dataset = TestDataset(self.spec.test_dataset)
        self.probabilities = self.load_probabilities()