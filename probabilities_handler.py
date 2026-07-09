from model_handler import Model
from model_handler import probabilities as legacy_probabilities
from base_model_handler import Base
from spec import ProbabilitySpec, HumanProbabilitySpec, _SEPARATOR
from datasets import Dataset
from test_dataset_handler import t, TestDataset
from dataset_handler import TrainingDataset, d
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from typing import cast, Sequence, Any, Iterable
from math import prod
from ctc_decoder import decode_probabilities
from pooling_handler import Pooling

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
        self.probabilities = self.load_probabilities(prefix=prefix, model_kwargs=model_kwargs)

    def pool(self, *args: str, **kwargs: str | Iterable[str]) -> pd.DataFrame:
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
                if not isinstance(value, str) and isinstance(value, Iterable):
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

    def load_probabilities(self, prefix: str = _PROBABILITIES_PREFIX, model_kwargs: dict[str, Any] = {}) -> pd.DataFrame:
        """Loads fully-prepared probabilities from a specification. Attempts to create them if missing."""

        try:
            probabilities = pd.read_csv(f'{self.path}{prefix}{_SEPARATOR}{self.spec}.csv')
        except FileNotFoundError:
            if 'max' in str(self.spec) or 'mean' in str(self.spec):
                model = Model(self.spec.model, **model_kwargs)
                label2id = model.vocab
                probabilities = legacy_probabilities(model.model, self.test_dataset.dataset, feature_extractor=Base(model.spec.base).feature_extractor, id2label = {v: k for k, v in label2id.items()})
            elif 'ctc' in str(self.spec) and 'vowel' in str(self.spec) or 'consonant' in str(self.spec):
                model = Model(self.spec.model, **model_kwargs)
                pooling = Pooling(self.spec.pooling)
                logits = self.test_dataset.dataset.map(model.as_map(), desc="Running model", batched=True, batch_size=32).with_format('torch')
                probabilities = cast(pd.DataFrame, logits.map(pooling.as_map(model), desc="Decoding logits", batched=True, batch_size=32, remove_columns=['logits', 'input_values']).to_pandas())
                probabilities['classification'] = probabilities['classification'].map(TrainingDataset(model.spec.output_dataset).get_translator(self.test_dataset.spec))
            else:
                raise NotImplementedError("Cannot dynamically generate probabilities yet.")
        
        probabilities = self.prepare_probabilities(probabilities)
        return probabilities


    def prepare_probabilities(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Prepares probabilities from the file."""

        probabilities = sorting_fn[self.spec.test_dataset.value](probabilities)
        if isinstance(self.spec, ProbabilitySpec):
            training_dataset = self.spec.model.output_dataset
            if training_dataset.family == self.spec.test_dataset.value:
                probabilities = self._add_training(probabilities)
            else:
                probabilities['training'] = 'no'
        return probabilities

    def _add_training(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Adds a column indicating which files were present in the training dataset for the model."""

        training_dataset = d[self.spec.model.last_training.training_dataset]
        train_ds = training_dataset.get_split(self.spec.model.training_split)
        test_ds = training_dataset.get_split(self.spec.model.eval_split)
        if not f'{self.test_dataset.spec}_file' in train_ds.column_names and 'input_values' in train_ds.column_names:
            d[self.spec.model.last_training.training_dataset].add_files(self.test_dataset)
            training_dataset = d[self.spec.model.last_training.training_dataset]
            train_ds = training_dataset.get_split(self.spec.model.training_split)
            test_ds = training_dataset.get_split(self.spec.model.eval_split)
        probabilities['training'] = probabilities['file'].case_when(caselist=[(probabilities['file'].isin(train_ds[f'{self.test_dataset.spec}_file']), 'train'), (probabilities['file'].isin(test_ds[f'{self.test_dataset.spec}_file']), 'test'), (pd.Series(True, index=probabilities.index), 'no')])
        return probabilities
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(str(self.spec))})'

vowel_order = 'iɪyʏeɛøœaæɐɑʌoɔɤuʊɯ:ː\u0303'
consonant_order = 'ptkbdɡhfθsʃvðzʒtʃdʒmnŋrljwy'

def world_vowel_sort(data: pd.DataFrame):
    data = data.sort_values(
        by = 'classification',
        key = lambda x: pd.Series([vowel_order.find(c) for c in s] for s in x),
    ).sort_values(by = 'file', kind = 'mergesort').sort_values(
        by = 'vowel',
        key = lambda x: pd.Series([vowel_order.find(c) for c in s] for s in x),
        kind = 'mergesort'
    ).sort_values(by = ['language'], kind = 'mergesort')
    return data

def consonant_challenge_sort(data: pd.DataFrame):
    data = data.sort_values(
        by = 'classification',
        key = lambda x: pd.Series([consonant_order.find(c) for c in s] for s in x),
    ).sort_values(by = 'file', kind = 'mergesort').sort_values(
        by = 'consonant',
        key = lambda x: pd.Series([consonant_order.find(c) for c in s] for s in x),
        kind = 'mergesort'
    )
    return data

sorting_fn = {'wv': world_vowel_sort, 'cc': consonant_challenge_sort}

class HumanProbabilities(Probabilities):
    spec: HumanProbabilitySpec

    def __init__(self, spec: str | HumanProbabilitySpec, path: str = _PROBABILITIES_PATH):
        self.spec = HumanProbabilitySpec(spec)
        self.path = path
        self.test_dataset = TestDataset(self.spec.test_dataset)
        self.probabilities = self.load_probabilities()

    def pool(self, *args: str, **kwargs: str | Iterable[str]) -> pd.DataFrame:
        """Ignore column 'training'."""

        if 'training' in kwargs:
            kwargs.pop('training')
        return super().pool(*args, **kwargs)