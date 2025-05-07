from datasets import Dataset, Audio#, load_dataset
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC #, TrainingArguments, Trainer
import evaluate
import torch
from dataclasses import dataclass
# from typing import List, Dict, Optional, Union
import numpy, pandas
from pyctcdecode import build_ctcdecoder
import random
import seaborn
import matplotlib.pyplot as plt

from model_handler import centre_probabilities, select_where, count_where, probability, pool, map_to_result, map_to_result_no_labels

human_responses = pandas.read_csv('../human_vowel_responses.csv')
human_responses = human_responses[human_responses['language_indiv'] == 'english'].rename(columns = {'#phone': 'phone'})
audio_files = list(set(human_responses['filename']))
languages = [list(set(human_responses[human_responses['filename'] == audio_file]['language_stimuli'])) for audio_file in audio_files]
for i, language in enumerate(languages):
    assert len(language) == 1
    languages[i] = language[0]

vowels = [list(set(human_responses[human_responses['filename'] == audio_file]['phone'])) for audio_file in audio_files]
for i, vowel in enumerate(vowels):
    assert len(vowel) == 1
    vowels[i] = vowel[0]

world_vowels = Dataset.from_dict({'audio': [f'../stimuli_world_vowels/{audio_file}.wav' for audio_file in audio_files], 'language': languages, 'vowel': vowels, 'file': audio_files}).cast_column('audio', Audio())
timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
vowel_order = 'iɪyʏeɛøœaæɐɑʌoɔɤuʊɯ:ː\u0303'
possible_human_responses = sorted(list(set(human_responses['assimilation'])), key = lambda x: [vowel_order.index(c) for c in x])



world_vowels = world_vowels.map(centre_probabilities)



# vowels_languages = [(vowel, language) for vowel in sorted(list(set(vowels)), key = lambda x: [vowel_order.index(c) for c in x]) for language in sorted(list(set(human_responses[human_responses['phone'] == vowel]['language_stimuli'])))]
# human_responses_pooled = pandas.DataFrame({
#     'vowel.language': [f'{vowel}.{language}' for vowel, language in vowels_languages],
#     **{response: [count_where(human_responses, language_indiv = 'english', phone = vowel, language_stimuli = language, assimilation = response)/count_where(human_responses, language_indiv = 'english', phone = vowel, language_stimuli = language) for vowel, language in vowels_languages] for response in possible_human_responses}})

# def heatmap(df, language = None, vowel = None, display = True):
#     df = df[df['vowel.language'].str.contains(f'{vowel or ""}.{language or ""}')].set_index('vowel.language')
#     seaborn.heatmap(df, xticklabels = True, vmin = 0, vmax = 1, cmap = 'crest', square = True)
#     if display:
#         plt.show()

# def human_heatmap(language = None, vowel = None, display = True):
#     heatmap(human_responses_pooled, language = language, vowel = vowel, display = display)

# model_probabilities_pooled = pool(world_vowels, vocab_list[:-1], vowels_languages, ['vowel', 'language'])
# model_probabilities_pooled = model_probabilities_pooled.reset_index(names = 'vowel.language')
# def model_heatmap(language = None, vowel = None, display = True):
#     heatmap(model_probabilities_pooled, language = language, vowel = vowel, display = display)

# model_vowel_probabilities_pooled = pool(world_vowels, timit_vowels, vowels_languages, ['vowel', 'language'])
# def model_vowel_heatmap(language = None, vowel = None):
#     heatmap(model_vowel_probabilities_pooled, language = language, vowel = vowel)

human_timit_vowels = {'i': 'iy', 'ɪ': 'ih', 'eɪ': 'ey', 'ɛ': 'eh', 'æ': 'ae', 'ɑ': 'aa', 'ʌ': 'ah', 'oʊ': 'ow', 'u': 'uw', 'ʊ': 'uh'}
# model_human_vowel_probabilities_pooled = pool(world_vowels, human_timit_vowels, vowels_languages, ['vowel', 'language'])
# model_human_vowel_probabilities_pooled = model_human_vowel_probabilities_pooled.reset_index(names = 'vowel.language')
# def model_human_vowel_heatmap(language = None, vowel = None, display = True):
#     heatmap(model_human_vowel_probabilities_pooled, language = language, vowel = vowel, display = display)

formants = pandas.read_csv('../stimuli_world_vowels/formants/world_vowels_formants.csv')
formants = formants.set_index('file')

model_human_vowel_probabilities_formants = pool(world_vowels, human_timit_vowels, audio_files, 'file')
model_human_vowel_probabilities_formants = pandas.concat([model_human_vowel_probabilities_formants, formants], axis = 1)

# def sample(probabilities, n = 1000):
#     frequencies = probabilities
#     responses = [i for i in probabilities.columns if i[0] != 'F']

#     for i, row in probabilities.iterrows():
#         print(sum(row[~probabilities.columns.str.contains('F')]))
#         # print(frequencies.loc[i,~frequencies.columns.str.contains('F')])
#         # print([row[response] for response in responses])
#         # print(numpy.unique(numpy.random.choice(responses, size = n, p = [row[response] for response in responses]), return_counts = True))
#         # temp, frequencies.loc[i,~frequencies.columns.str.contains('F')] = numpy.unique(numpy.random.choice(responses, size = n, p = [row[response] for response in responses]), return_counts = True)

#     return frequencies

# sample(model_human_vowel_probabilities_formants)

# human_responses['F1'] = human_responses.apply(lambda x: formants.loc[x['filename'], 'F1'], axis = 1)
# human_responses['F2'] = human_responses.apply(lambda x: formants.loc[x['filename'], 'F2'], axis = 1)
# human_responses['F3'] = human_responses.apply(lambda x: formants.loc[x['filename'], 'F3'], axis = 1)

# human_responses['F1_norm'] = human_responses['F1'] / human_responses['F3']
# human_responses['F2_norm'] = human_responses['F2'] / human_responses['F3']

# human_responses['F1_norm_jitter'] = human_responses['F1'] / human_responses['F3'] + numpy.random.randn(len(human_responses['F1']))*0.005
# human_responses['F2_norm_jitter'] = human_responses['F2'] / human_responses['F3'] + numpy.random.randn(len(human_responses['F1']))*0.005

# seaborn.scatterplot(data = human_responses, x = 'F1_norm_jitter', y = 'F2_norm_jitter', hue = 'assimilation', size = 400, alpha = 0.5, hue_order = human_timit_vowels.keys(), palette = "Set3")
# plt.show()

model_human_vowel_probabilities_formants['F1_norm'] = model_human_vowel_probabilities_formants['F1'] / model_human_vowel_probabilities_formants['F3']
model_human_vowel_probabilities_formants['F2_norm'] = model_human_vowel_probabilities_formants['F2'] / model_human_vowel_probabilities_formants['F3']

human_responses_formants = pandas.DataFrame({
    'file': [filename for filename in audio_files],
    **{response: [count_where(human_responses, language_indiv = 'english', filename = filename, assimilation = response)/count_where(human_responses, language_indiv = 'english', filename = filename) for filename in audio_files] for response in possible_human_responses},
    'F1': [formants.loc[filename, 'F1'] for filename in audio_files], 'F2': [formants.loc[filename, 'F2'] for filename in audio_files], 'F3': [formants.loc[filename, 'F3'] for filename in audio_files]})

human_responses_formants['F1_norm'] = human_responses_formants['F1'] / human_responses_formants['F3']
human_responses_formants['F2_norm'] = human_responses_formants['F2'] / human_responses_formants['F3']


for file in audio_files:
    vowel = list(human_responses[human_responses['filename'] == file]['phone'])[0]
    language = list(human_responses[human_responses['filename'] == file]['language_stimuli'])[0]

    human_responses_formants.loc[human_responses_formants['file'] == file, 'vowel.language'] = f'{vowel}.{language}'
    model_human_vowel_probabilities_formants.loc[file, 'vowel.language'] = f'{vowel}.{language}'

for i, response in enumerate(['u']):#enumerate(human_timit_vowels.keys()):
    color = seaborn.husl_palette(10)[i]
    seaborn.scatterplot(data = human_responses_formants, x = 'F1_norm', y = 'F2_norm', hue = response, style = 'vowel.language', markers = [f'${i}$' for i in human_responses_formants['vowel.language'].unique()], style_order = list(human_responses_formants['vowel.language'].unique()), size = response, sizes = (10, 1500), alpha = 0.6, palette = seaborn.light_palette(color, as_cmap = True))
    plt.figure()
    seaborn.scatterplot(data = model_human_vowel_probabilities_formants, x = 'F1_norm', y = 'F2_norm', hue = response, style = 'vowel.language', markers = [f'${i}$' for i in human_responses_formants['vowel.language'].unique()], style_order = list(human_responses_formants['vowel.language'].unique()), size = response, sizes = (10, 1500), alpha = 0.6, palette = seaborn.light_palette(color, as_cmap = True))
    plt.show()