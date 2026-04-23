# %%
from util import ProbabilitiesMap
import pandas as pd
import datasets
from scipy.spatial.distance import jensenshannon
from probabilities_handler import vowel_order
p = ProbabilitiesMap()

# %%[markdown]
# # Experiment 1
# ## Get relevant models, calculate metrics
# %%
probs = [prob for prob in p.find_probabilities() if 'cross' in prob and 'w2v2_' in prob and 'ctc' not in prob and 'wvENResponses10Fold' in prob and 'cross_N' not in prob]
# label folds with 
folds = datasets.load_from_disk('../prep_wvENResponses10Fold')
test_df = p[probs[0]].test_dataset.dataset.to_pandas()
test_df['input_values'] = test_df['input_values'].apply(tuple)
folds = folds.map(lambda x: {'file': test_df['file'][test_df['input_values'] == tuple(x['input_values'])].item()})
label_order = sorted(list(range(10)), key=lambda i: [vowel_order.index(c) for c in folds['fold_0'].features['label'].names[i]])
folds_df = [folds[f'fold_{i}'].to_pandas().pivot_table(columns='label', index='file', aggfunc='count', fill_value=0)['input_values'][label_order] for i in range(10)]
folds_df = [fold_df.apply(lambda x: x / x.sum(), axis = 1) for fold_df in folds_df]
# %%
data = {'model_var': [], 'model_cross': [],
    'JS-native': [], 'JS-nonnative': [],
    'acc-native': [], 'acc-nonnative': [],
}
for prob in probs:
    model_fold = int(prob[-4:-3])
    data['model_var'].append(prob[:prob.find('_cross')])
    data['model_cross'].append(model_fold)
    df_excl = p[prob].pool('file', language='EN')
    js_divs_excl = [jensenshannon(fold_row[1].to_numpy(), p_row[1].to_numpy())**2 for fold_row, p_row in zip(df_excl.loc[folds_df[model_fold].index].iterrows(), folds_df[model_fold].iterrows())]
    js_divs_excl = pd.Series(js_divs_excl)
    data['JS-native'].append(js_divs_excl.mean())
    data['JS-nonnative'].append(pd.concat(pd.Series(i) for i in [p.js_divergence(prob, 'humans_wv', 'file', language=lang) for lang in ['BR', 'ES', 'FR', 'GL', 'GR', 'TU']]).mean())
    correct_excl = folds[f'fold_{model_fold}'].map(lambda x: {'correct': (df_excl.loc[x['file']].argmax() == label_order.index(x['label']))})['correct']
    data['acc-native'].append(len([i for i in correct_excl if i])/len(correct_excl))
    correct_nonnative = pd.concat([p['humans_wv'].pool('file', language=lang) for lang in ['BR', 'ES', 'FR', 'GL', 'GR', 'TU']], axis = 0).apply(lambda x: x.argmax() == p[prob].pool('file').loc[x.name].argmax(), axis = 1)
    data['acc-nonnative'].append(len(correct_nonnative[correct_nonnative])/len(correct_nonnative))

df = pd.DataFrame(data)

    
# %%[markdown]
# # Experiment 2
# ## Get relevant models, calculate metrics
# %%
probs = [prob for prob in p.find_probabilities() if 'cross' in prob and ('w2v2_' in prob or 'w2v2fr_' in prob or 'mfcc_' in prob) and 'ctc' not in prob and 'max' in prob and 'relu' not in prob and 'wvENResponses10Fold' in prob and 'cross_N' not in prob and 'var' not in prob]
# label folds with 
folds = datasets.load_from_disk('../prep_wvENResponses10Fold')
test_df = p[probs[0]].test_dataset.dataset.to_pandas()
test_df['input_values'] = test_df['input_values'].apply(tuple)
folds = folds.map(lambda x: {'file': test_df['file'][test_df['input_values'] == tuple(x['input_values'])].item()})
label_order = sorted(list(range(10)), key=lambda i: [vowel_order.index(c) for c in folds['fold_0'].features['label'].names[i]])
folds_df = [folds[f'fold_{i}'].to_pandas().pivot_table(columns='label', index='file', aggfunc='count', fill_value=0)['input_values'][label_order] for i in range(10)]
folds_df = [fold_df.apply(lambda x: x / x.sum(), axis = 1) for fold_df in folds_df]
# %%
data = {'model_var': [], 'model_cross': [],
    'JS-native': [], 'JS-nonnative': [],
    'acc-native': [], 'acc-nonnative': [],
}
for prob in probs:
    model_fold = int(prob[-4:-3])
    data['model_var'].append(prob[:prob.find('_cross')])
    data['model_cross'].append(model_fold)
    df_excl = p[prob].pool('file', language='EN')
    js_divs_excl = [jensenshannon(fold_row[1].to_numpy(), p_row[1].to_numpy())**2 for fold_row, p_row in zip(df_excl.loc[folds_df[model_fold].index].iterrows(), folds_df[model_fold].iterrows())]
    js_divs_excl = pd.Series(js_divs_excl)
    data['JS-native'].append(js_divs_excl.mean())
    data['JS-nonnative'].append(pd.concat(pd.Series(i) for i in [p.js_divergence(prob, 'humans_wv', 'file', language=lang) for lang in ['BR', 'ES', 'FR', 'GL', 'GR', 'TU']]).mean())
    correct_excl = folds[f'fold_{model_fold}'].map(lambda x: {'correct': (df_excl.loc[x['file']].argmax() == label_order.index(x['label']))})['correct']
    data['acc-native'].append(len([i for i in correct_excl if i])/len(correct_excl))
    correct_nonnative = pd.concat([p['humans_wv'].pool('file', language=lang) for lang in ['BR', 'ES', 'FR', 'GL', 'GR', 'TU']], axis = 0).apply(lambda x: x.argmax() == p[prob].pool('file').loc[x.name].argmax(), axis = 1)
    data['acc-nonnative'].append(len(correct_nonnative[correct_nonnative])/len(correct_nonnative))

df2 = pd.DataFrame(data)

    
# %%
