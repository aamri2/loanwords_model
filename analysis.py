# %%
from util import ProbabilitiesMap
import datasets
from test_dataset_handler import TestDataset
from probabilities_handler import Probabilities, vowel_order
from typing import Callable, Iterable, cast
from spec import ModelSpec, ProbabilitySpec, HumanProbabilitySpec
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from tqdm.notebook import tqdm

ALL_LANGUAGES = {'BR', 'ES', 'FR', 'GL', 'GR', 'TU', 'EN'}

def compute_metrics(p: ProbabilitiesMap) -> pd.DataFrame:
    """Compute JS divergence and accuracy metrics with the correct human responses for all model probabilities in p."""

    all_probs = p.find_probabilities()
    probs = [] # exclude human responses
    for prob in all_probs: # exclude human responses
        try:
            ProbabilitySpec(prob)
            probs.append(prob)
        except ValueError:
            HumanProbabilitySpec(prob) # make sure failed because human, not because of logic error
    data = {
        'model_var': [], 'model_cross': [],
        'JS-native': [], 'JS-nonnative': [],
        'acc-native': [], 'acc-nonnative': [],
    }

    for prob in tqdm(probs, desc='Computing metrics'):
        data['model_var'].append(prob[:prob.find('_cross') if 'cross' in prob else prob.find('_wv')]) # {...}_cross
        data['model_cross'].append(int(prob[prob.find('cross_') + len('cross_')]) if 'cross' in prob else None) # cross_{X}
        if p[prob].spec.model.output_dataset.language:
            if p[prob].spec.model.last_training.training_dataset.language == 'EN': # default to EN
                native = 'EN'
                nonnative = ALL_LANGUAGES.difference(['EN'])
                humans = 'humans_wv'
            elif p[prob].spec.model.output_dataset.language == 'FR':
                native = 'FR'
                nonnative = ALL_LANGUAGES.difference(['FR'])
                humans = 'humansFR_wv'
            else:
                raise NotImplementedError('Only configured for EN or FR right now.')
        else:
            if p[prob].spec.model.output_dataset.family == 'timit':
                native = 'EN'
                nonnative = ALL_LANGUAGES.difference(['EN'])
                humans = 'humans_wv'
            elif p[prob].spec.model.output_dataset.family == 'bl':
                native = 'FR'
                nonnative = ALL_LANGUAGES.difference(['FR'])
                humans = 'humansFR_wv'
            else:
                raise NotImplementedError(f"Can't infer language of dataset {p[prob].spec.model.output_dataset}.")
        # native only on held-out data
        js_native = p.js_divergence(humans, prob, 'file', language=native, training=['test', 'no'])
        data['JS-native'].append(js_native.mean())
        js_nonnative = p.js_divergence(humans, prob, 'file', language=nonnative)
        data['JS-nonnative'].append(js_nonnative.mean())

        acc_native = p.accuracy(humans, prob, 'file', language=native, training=['test', 'no'])
        data['acc-native'].append(acc_native)
        acc_nonnative = p.accuracy(humans, prob, 'file', language=nonnative)
        data['acc-nonnative'].append(acc_nonnative)

    return pd.DataFrame(data)

def compute_benchmarks(datasets: dict[str, datasets.DatasetDict]) -> pd.DataFrame:
    """
    Compute benchmarks on 10-fold prepared datasets containing WV stimuli.
    """
    
    wv = cast(pd.DataFrame, TestDataset('wv').dataset.to_pandas())
    wv['input_values'] = wv['input_values'].map(tuple)
    metrics = {'dataset': [], 'accuracy_ceil': [], 'mean_js_div': [], 'js_div_se': []}
    for name, dataset in datasets.items():
        df = pd.concat((cast(pd.DataFrame, dataset[f'fold_{i}'].to_pandas()) for i in range(10)), keys=range(10)).reset_index(level=0, names='fold')
        if 'file' not in df.columns or 'vowel' not in df.columns:
            df['input_values'] = df['input_values'].map(tuple)
            df = df.merge(wv) # add metadata
        # calculate accuracy ceiling
        modes = df.groupby('file')['label'].agg(lambda x: x.value_counts().index[0]).rename('mode')
        df = df.merge(modes, on='file') # add modal labels
        accuracy = (df['label'] == df['mode']).astype(int).mean()
        distributions =  df.pivot_table(columns='label', index=['fold', 'vowel_language' if 'vowel_language' in df.columns else 'vowel'], aggfunc='size', fill_value=0).apply(lambda x: x/x.sum())
        js_divs = []
        for fold in range(10):
            train_distributions = distributions.xs(fold, level=0)
            test_distributions = distributions[distributions.index.get_level_values(0) != fold].groupby('vowel_language' if 'vowel_language' in df.columns else 'vowel').mean()
            js_divs.append(jensenshannon(train_distributions, test_distributions.loc[train_distributions.index], axis=1).mean())
        js_divs = pd.Series(js_divs)
        metrics['dataset'].append(name)
        metrics['accuracy_ceil'].append(accuracy)
        metrics['mean_js_div'].append(js_divs.mean())
        metrics['js_div_se'].append(js_divs.std() / (len(js_divs)**.5))
    return pd.DataFrame(metrics)

def assimilation_heatmaps(p: ProbabilitiesMap, title_generator:Callable[[ModelSpec], str]|None=None):
    """
    Heatmaps for each language for each model variant, with the native language first.
    
    Titles generated automatically as the model spec, but a function to transform
    model specs into readable titles may be provided.
    """
    
    model_vars = {prob[:prob.find('_cross')] for prob in p.find_probabilities() if type(p[prob]) is Probabilities}
    
    for model_var in model_vars:
        language_order = sorted(ALL_LANGUAGES)
        if 'FR' or 'bl' in model_var:
            language_order.remove('FR')
            language_order = ['FR'] + language_order
        else:
            language_order.remove('EN')
            language_order = ['EN'] + language_order
        dfs = {language: pd.concat((p[prob].pool('vowel', language=language, training=['test', 'no']) for i in range(10) for prob in [f'{model_var}_cross_{i}_wv'])) for language in ALL_LANGUAGES}
        dfs = {language: df.groupby('vowel').mean().sort_index(key=lambda x: [[vowel_order.index(c) for c in s] for s in x]) for language, df in dfs.items()}
        df = pd.concat((dfs[language] for language in language_order), keys=language_order).reset_index(level=0, names='language')

        prob_cols = [col for col in df.columns if col != 'language']

        blocks = []
        xticklabels = []

        for lang, group in df.groupby('language', sort=False):
            # keep the original row labels for the heatmap
            blocks.append(group[prob_cols])

            # original tick labels (whatever your DataFrame index is)
            xticklabels.extend(group.index.astype(str).tolist())

            # separator row between language groups
            blocks.append(pd.DataFrame(np.nan, index=[''], columns=prob_cols))
            xticklabels.append('')  # blank tick for separator

        # remove trailing separator
        combined = pd.concat(blocks[:-1])
        xticklabels = xticklabels[:-1]

        ax = sns.heatmap(
            combined.T,
            square=True,
            cmap='crest',
            xticklabels=xticklabels,
            yticklabels=True,
            vmin=0,
            vmax=1,
        )

        # Add centered language labels below the existing tick labels
        start = 0
        for lang, group in df.groupby('language', sort=False):
            n = len(group)
            center = start + n / 2

            ax.text(
                center,
                1.2,
                lang,
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform()
            )

            start += n + 1  # account for separator row
        plt.title(model_var if not title_generator else title_generator(ModelSpec(model_var)))
        plt.show()

# %%
p1 = ProbabilitiesMap(path='probabilities/experiment_1/')
df1 = compute_metrics(p1)
# %%
df1['arch'] = df1['model_var'].apply(lambda x: 'mean' if 'mean' in x else 'max' if 'max' in x else 'knn' if 'knn' in x else None)
df1['rep'] = df1['model_var'].apply(lambda x: x[:x.find('_')])
df1['lang'] = df1['model_var'].apply(lambda x: 'EN' if 'EN' in x else 'FR' if 'FR' in x else None)
df1['domain'] = df1['model_var'].apply(lambda x: 'non-nat' if 'Nonnative' in x else 'nat')
df1[[col for col in df1.columns if 'model' not in col]]\
    .groupby(['lang', 'domain', 'arch', 'rep'])\
    .agg(['mean', 'sem'])\
    .xs('EN').xs('nat').xs('formant', level=1)\
    .style.highlight_min(subset=pd.IndexSlice[:, pd.IndexSlice[:'JS-nonnative', 'mean']]).highlight_max(subset=pd.IndexSlice[:, pd.IndexSlice['acc-native':, 'mean']])
# %%
p2 = ProbabilitiesMap(path='probabilities/experiment_2/')
df2 = compute_metrics(p2)
# %%
df2['type'] = df2['model_var'].apply(lambda x: 'gold' if 'wv' in x else 'pseudo-sylls' if 'EV' in x else 'ASR' if 'ctc' in x else None)
df2['lang'] = df2['model_var'].apply(lambda x: 'EN' if 'w2v2-large' in x else 'FR' if 'w2v2fr-large' in x else None)
df2[[col for col in df2.columns if 'model' not in col]]\
    .groupby(['lang', 'type'])\
    .agg(['mean', 'sem'])\
    .style.highlight_min(subset=pd.IndexSlice[:, pd.IndexSlice[:'JS-nonnative', 'mean']]).highlight_max(subset=pd.IndexSlice[:, pd.IndexSlice['acc-native':, 'mean']])
