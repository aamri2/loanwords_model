from spec import ProbabilitySpec, HumanProbabilitySpec, _SEPARATOR
from datasets import Dataset
from test_dataset_handler import t, TestDataset
from dataset_handler import TrainingDataset
from model_handler import Model
from model_handler import probabilities as legacy_probabilities
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from typing import cast, Sequence, Any
from math import prod

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

    def __init__(self, spec: str | ProbabilitySpec, path: str = _PROBABILITIES_PATH, prefix: str = _PROBABILITIES_PREFIX, model_kwargs: dict[str, Any] = {}):
        self.spec = ProbabilitySpec(spec)
        self.path = path
        self.test_dataset = t[self.spec.test_dataset]
        self.model = Model(self.spec.model, **model_kwargs)
        self.probabilities = self.load_probabilities(prefix=prefix)

    def pool(self, *args: str, **kwargs: str) -> pd.DataFrame:
        """
        Returns a pivot table in wide format.
        
        Args:
            *args: Columns to pool across.
            **kwargs: Columns to filter by.
        """
        probabilities = self.probabilities

        if kwargs:
            mask = pd.Series(True, index=probabilities.index)
            for column, value in kwargs.items():
                if not isinstance(value, str) and isinstance(value, Sequence):
                    mask = mask & probabilities[column].isin(value)
                else:
                    mask = mask & (probabilities[column] == value)
            probabilities = probabilities.loc[mask]
        
        pooled_probabilities = probabilities.pivot_table(values = 'probabilities', columns = 'classification', index = args, sort = False)
        return pooled_probabilities
    
    def heatmap(self, *args: str, **kwargs: str):
        """Uses pool to create a seaborn heatmap."""

        plt.figure()
        sns.heatmap(self.pool(*args, **kwargs), cmap = 'crest', square = True)
        plt.title(str(self.spec), wrap=True)
        plt.show(block = False)
    
    def entropy(self, *args: str, **kwargs: str) -> np.ndarray:
        entropies = stats.entropy(self.pool(*args, **kwargs), axis = 1)
        return cast(np.ndarray, entropies)

    def entropy_histogram(self, *args: str, **kwargs: str):
        plt.figure()
        entropy = self.entropy(*args, **kwargs)
        sns.histplot(entropy)
        plt.axvline(x = entropy.mean(), linestyle = 'dashed', linewidth = 1)
        plt.text(entropy.mean()*1.1, plt.ylim()[1]*0.9, f'Mean: {entropy.mean():.2f}')
        plt.title(f'{self.spec} entropies (by {args}, {kwargs})', wrap=True)
        plt.show(block = False)

    def load_probabilities(self, prefix: str = _PROBABILITIES_PREFIX) -> pd.DataFrame:
        """Loads fully-prepared probabilities from a specification. Attempts to create them if missing."""

        try:
            probabilities = pd.read_csv(f'{self.path}{prefix}{_SEPARATOR}{self.spec}.csv')
        except:
            if 'max' in str(self.spec) or 'mean' in str(self.spec):
                label2id = self.model.vocab
                probabilities = legacy_probabilities(self.model.model, self.test_dataset.dataset, id2label = {v: k for k, v in label2id.items()})
            else:
                raise NotImplementedError("Cannot dynamically generate probabilities yet.")
        
        probabilities = self.prepare_probabilities(probabilities)
        return probabilities


    def prepare_probabilities(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Prepares probabilities from the file."""

        probabilities = self._add_formants(probabilities)
        probabilities = self._add_contexts(probabilities)
        probabilities = world_vowel_sort(probabilities)
        if isinstance(self.spec, ProbabilitySpec):
            training_dataset = self.spec.model.training[-1].training_dataset if isinstance(self.spec.model.training, tuple) else self.spec.model.training.training_dataset
            if training_dataset.family == self.spec.test_dataset.value:
                probabilities = self._add_training(probabilities)
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

    def _add_training(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Adds a column indicating which files were present in the training dataset for the model."""

        training_dataset = TrainingDataset(self.spec.model.training[-1].training_dataset if isinstance(self.spec.model.training, tuple) else self.spec.model.training.training_dataset)
        train_df = cast(pd.DataFrame, training_dataset.get_split(self.model.training_split).to_pandas())
        test_df = cast(pd.DataFrame, training_dataset.get_split(self.model.eval_split).to_pandas())
        if not 'file' in train_df.columns and 'input_values' in train_df.columns:
            train_df = self._add_files(train_df)
            test_df = self._add_files(train_df)
        probabilities['training'] = probabilities['file'].case_when(caselist=[(probabilities['file'].isin(train_df['file']), 'train'), (probabilities['file'].isin(test_df['file']), 'test'), (pd.Series(True, index=probabilities.index), 'no')])
        return probabilities
    
    def _add_files(self, df: pd.DataFrame):
        """Add file names to a DataFrame containing input_values."""

        test_df = cast(pd.DataFrame, self.test_dataset.dataset.to_pandas())
        input_values = test_df['input_values'].apply(tuple)
        df['file'] = df.apply(lambda row: next(iter(test_df['file'][input_values == tuple(row['input_values'])]), None), axis=1) # allow for unrecognized files
        return df
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(str(self.spec))})'

vowel_order = 'iɪyʏeɛøœaæɐɑʌoɔɤuʊɯ:ː\u0303'
def world_vowel_sort(data: pd.DataFrame):
    data = data.sort_values(
        by = 'classification',
        key = lambda x: pd.Series([vowel_order.index(c) for c in s] for s in x),
    ).sort_values(by = 'file', kind = 'mergesort').sort_values(
        by = 'vowel',
        key = lambda x: pd.Series([vowel_order.index(c) for c in s] for s in x),
        kind = 'mergesort'
    ).sort_values(by = ['language'], kind = 'mergesort')
    return data

class HumanProbabilities(Probabilities):
    spec: HumanProbabilitySpec

    def __init__(self, spec: str | HumanProbabilitySpec):
        self.spec = HumanProbabilitySpec(spec)
        self.test_dataset = TestDataset(self.spec.test_dataset)
        self.probabilities = self.load_probabilities()