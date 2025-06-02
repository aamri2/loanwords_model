from datasets import Dataset, Audio#, load_dataset
import numpy, pandas
import random
import seaborn
import matplotlib.pyplot as plt
import json

from transformers import Wav2Vec2ForCTC
from train_sequence_classification import Wav2Vec2WithAttentionClassifier
from model_handler import heatmap, pool, model, probabilities, audio_to_input_values, feature_extractor, ctc_wrapper

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

vowel_order = 'iɪyʏeɛøœaæɐɑʌoɔɤuʊɯ:ː\u0303'

def world_vowel_sort(data: pandas.DataFrame):
    data = data.sort_values(
        by = 'classification',
        key = lambda x: [[vowel_order.index(c) for c in s] for s in x]
    ).sort_values(by = 'file', kind = 'mergesort').sort_values(
        by = 'vowel',
        key = lambda x: [[vowel_order.index(c) for c in s] for s in x],
        kind = 'mergesort'
    ).sort_values(by = ['language'], kind = 'mergesort')
    return data

# classification model
try:
    world_vowel_probabilities = pandas.read_csv('probabilities/world_vowels_classification.csv')
except FileNotFoundError:
    world_vowels = Dataset.from_dict({'audio': [f'../stimuli_world_vowels/{audio_file}.wav' for audio_file in audio_files], 'language': languages, 'vowel': vowels, 'file': audio_files}).cast_column('audio', Audio())
    world_vowels = audio_to_input_values(world_vowels, feature_extractor)
    world_vowel_probabilities = probabilities(model, world_vowels)
    world_vowel_probabilities = world_vowel_sort(world_vowel_probabilities)
    world_vowel_probabilities.to_csv('probabilities/world_vowels_classification.csv')


# masked classification model
try:
    world_vowel_masked_probabilities = pandas.read_csv('probabilities/world_vowels_masked_classification.csv')
except FileNotFoundError:
    masked_classification_model = Wav2Vec2WithAttentionClassifier.from_pretrained('../models/final_model_masked_classification')
    world_vowels = Dataset.from_dict({'audio': [f'../stimuli_world_vowels/{audio_file}.wav' for audio_file in audio_files], 'language': languages, 'vowel': vowels, 'file': audio_files}).cast_column('audio', Audio())
    world_vowels = audio_to_input_values(world_vowels, feature_extractor)
    world_vowel_masked_probabilities = probabilities(masked_classification_model, world_vowels)
    world_vowel_masked_probabilities = world_vowel_sort(world_vowel_masked_probabilities)
    world_vowel_masked_probabilities.to_csv('probabilities/world_vowels_masked_classification.csv')

timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
possible_human_responses = sorted(list(set(human_responses['assimilation'])), key = lambda x: [vowel_order.index(c) for c in x])
human_timit_vowels = {'i': 'iy', 'ɪ': 'ih', 'eɪ': 'ey', 'ɛ': 'eh', 'æ': 'ae', 'ɑ': 'aa', 'ʌ': 'ah', 'oʊ': 'ow', 'u': 'uw', 'ʊ': 'uh'}
timit_human_vowels = {value: key for key, value in human_timit_vowels.items()}

# human responses
try:
    human_responses_pooled = pandas.read_csv('probabilities/world_vowels_human.csv')
except FileNotFoundError:
    target_columns = {'language_stimuli': 'language', 'phone': 'vowel', 'filename': 'file'}
    human_responses_pooled = pandas.melt(pandas.crosstab([human_responses[column] for column in target_columns.keys()], human_responses['assimilation'], colnames = ['classification'], normalize = 'index'), ignore_index = False, value_name = 'probabilities').reset_index().rename(target_columns, axis = 'columns')
    human_responses_pooled = world_vowel_sort(human_responses_pooled)
    human_responses_pooled.to_csv('probabilities/world_vowels_human.csv')

# CTC centre probabilities
try:
    world_vowels_ctc = pandas.read_csv('probabilities/world_vowels_ctc.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/final_model_folded')
    world_vowels = Dataset.from_dict({'audio': [f'../stimuli_world_vowels/{audio_file}.wav' for audio_file in audio_files], 'language': languages, 'vowel': vowels, 'file': audio_files}).cast_column('audio', Audio())
    world_vowels = audio_to_input_values(world_vowels, feature_extractor)
    with open('../models/final_model_folded/vocab.json') as f:
        vocab = json.load(f)
    id2label_ctc = {vocab[timit_vowel]: human_vowel for timit_vowel, human_vowel in timit_human_vowels.items()}
    world_vowels_ctc = probabilities(ctc_wrapper(ctc_model), world_vowels, id2label_ctc)
    world_vowels_ctc = world_vowel_sort(world_vowels_ctc)
    world_vowels_ctc.to_csv('probabilities/world_vowels_ctc.csv')


prev_phone = human_responses.set_index('filename')['prev_phone'].groupby('filename').first().rename_axis(index = 'file')
next_phone = human_responses.set_index('filename')['next_phone'].groupby('filename').first().rename_axis(index = 'file')

world_vowel_probabilities.insert(len(world_vowel_probabilities.columns), 'prev_phone', prev_phone.loc[world_vowel_probabilities['file']].reset_index()['prev_phone'])
human_responses_pooled.insert(len(human_responses_pooled.columns), 'prev_phone', prev_phone.loc[human_responses_pooled['file']].reset_index()['prev_phone'])
world_vowels_ctc.insert(len(world_vowels_ctc.columns), 'prev_phone', prev_phone.loc[world_vowels_ctc['file']].reset_index()['prev_phone'])
world_vowel_probabilities.insert(len(world_vowel_probabilities.columns), 'next_phone', next_phone.loc[world_vowel_probabilities['file']].reset_index()['next_phone'])
human_responses_pooled.insert(len(human_responses_pooled.columns), 'next_phone', next_phone.loc[human_responses_pooled['file']].reset_index()['next_phone'])
world_vowels_ctc.insert(len(world_vowels_ctc.columns), 'next_phone', next_phone.loc[world_vowels_ctc['file']].reset_index()['next_phone'])


formants = pandas.read_csv('../stimuli_world_vowels/formants/world_vowels_formants.csv')
formants = formants.set_index('file')

# model_human_vowel_probabilities_formants = pool(world_vowels, human_timit_vowels, audio_files, 'file')
# model_human_vowel_probabilities_formants = pandas.concat([model_human_vowel_probabilities_formants, formants], axis = 1)

# human_responses['F1'] = human_responses.apply(lambda x: formants.loc[x['filename'], 'F1'], axis = 1)
# human_responses['F2'] = human_responses.apply(lambda x: formants.loc[x['filename'], 'F2'], axis = 1)
# human_responses['F3'] = human_responses.apply(lambda x: formants.loc[x['filename'], 'F3'], axis = 1)

# human_responses['F1_norm'] = human_responses['F1'] / human_responses['F3']
# human_responses['F2_norm'] = human_responses['F2'] / human_responses['F3']

# human_responses['F1_norm_jitter'] = human_responses['F1'] / human_responses['F3'] + numpy.random.randn(len(human_responses['F1']))*0.005
# human_responses['F2_norm_jitter'] = human_responses['F2'] / human_responses['F3'] + numpy.random.randn(len(human_responses['F1']))*0.005

# model_human_vowel_probabilities_formants['F1_norm'] = model_human_vowel_probabilities_formants['F1'] / model_human_vowel_probabilities_formants['F3']
# model_human_vowel_probabilities_formants['F2_norm'] = model_human_vowel_probabilities_formants['F2'] / model_human_vowel_probabilities_formants['F3']