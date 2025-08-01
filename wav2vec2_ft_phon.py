from datasets import Dataset, Audio#, load_dataset
import numpy, pandas
import random
import seaborn
import matplotlib.pyplot as plt
import json

from transformers import Wav2Vec2ForCTC
from model_architecture import Wav2Vec2WithAttentionClassifier, Wav2Vec2ForCTCWithAttentionClassifier, Wav2Vec2ForCTCWithTransformer
from model_handler import heatmap, pool, mae, diffmap, feature_extractor, probabilities, audio_to_input_values, ctc_wrapper, ctc_cvcBeamSearch_wrapper

human_responses = pandas.read_csv('../human_vowel_responses.csv')
fr_human_responses = human_responses[human_responses['language_indiv'] == 'french'].rename(columns = {'#phone': 'phone'})
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

world_vowels = Dataset.from_dict({'audio': [f'../stimuli_world_vowels/{audio_file}.wav' for audio_file in audio_files], 'language': languages, 'vowel': vowels, 'file': audio_files}).cast_column('audio', Audio())
world_vowels = audio_to_input_values(world_vowels, feature_extractor)

timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
possible_human_responses = sorted(list(set(human_responses['assimilation'])), key = lambda x: [vowel_order.index(c) for c in x])
human_timit_vowels = {'i': 'iy', 'ɪ': 'ih', 'eɪ': 'ey', 'ɛ': 'eh', 'æ': 'ae', 'ɑ': 'aa', 'ʌ': 'ah', 'oʊ': 'ow', 'u': 'uw', 'ʊ': 'uh'}
timit_human_vowels = {value: key for key, value in human_timit_vowels.items()}
with open('human_bl_vowels.json') as f:
    human_bl_vowels = json.load(f)
bl_human_vowels = {value: key for key, value in human_bl_vowels.items()}
bl_consonants = ['n', 'b', 'k', 's', 'Z', 'v', 'j', 'm', 'w', 'g', 't', 'R', 'l', 'd', 'S', 'N', 'z', 'p', 'f']

# ctc classification model
try:
    p_w2v2_ctc_1_timit_attn_class_3_wvEN_wv = pandas.read_csv('probabilities/p_w2v2_ctc_1_timit_attn_class_3_wvEN_wv.csv')
except FileNotFoundError:
    m_w2v2_ctc_1_timit_attn_class_3_wvEN = Wav2Vec2ForCTCWithAttentionClassifier.from_pretrained('../models/m_w2v2_ctc_1_timit_attn_class_3_wvEN')
    p_w2v2_ctc_1_timit_attn_class_3_wvEN_wv = probabilities(m_w2v2_ctc_1_timit_attn_class_3_wvEN, world_vowels, {i: v for i, v in enumerate(['eɪ', 'i', 'oʊ', 'u', 'æ', 'ɑ', 'ɛ', 'ɪ', 'ʊ', 'ʌ'])})
    p_w2v2_ctc_1_timit_attn_class_3_wvEN_wv = world_vowel_sort(p_w2v2_ctc_1_timit_attn_class_3_wvEN_wv)
    p_w2v2_ctc_1_timit_attn_class_3_wvEN_wv.to_csv('probabilities/p_w2v2_ctc_1_timit_attn_class_3_wvEN_wv.csv')


# classification model
try:
    p_w2v2_attn_class_2_timitEV_wv = pandas.read_csv('probabilities/p_w2v2_attn_class_2_timitEV_wv.csv')
except FileNotFoundError:
    p_w2v2_attn_class_2_timitEV_wv = probabilities(model, world_vowels)
    p_w2v2_attn_class_2_timitEV_wv = world_vowel_sort(p_w2v2_attn_class_2_timitEV_wv)
    p_w2v2_attn_class_2_timitEV_wv.to_csv('probabilities/p_w2v2_attn_class_2_timitEV_wv.csv')


# masked classification model
try:
    p_w2v2_attn_class_2_timitMV_wv = pandas.read_csv('probabilities/p_w2v2_attn_class_2_timitMV_wv.csv')
except FileNotFoundError:
    masked_classification_model = Wav2Vec2WithAttentionClassifier.from_pretrained('../models/m_w2v2_attn_class_2_timitMV')
    p_w2v2_attn_class_2_timitMV_wv = probabilities(masked_classification_model, world_vowels)
    p_w2v2_attn_class_2_timitMV_wv = world_vowel_sort(p_w2v2_attn_class_2_timitMV_wv)
    p_w2v2_attn_class_2_timitMV_wv.to_csv('probabilities/p_w2v2_attn_class_2_timitMV_wv.csv')

# human responses
try:
    p_humans_wv = pandas.read_csv('probabilities/p_humans_wv.csv')
except FileNotFoundError:
    target_columns = {'language_stimuli': 'language', 'phone': 'vowel', 'filename': 'file'}
    p_humans_wv = pandas.melt(pandas.crosstab([human_responses[column] for column in target_columns.keys()], human_responses['assimilation'], colnames = ['classification'], normalize = 'index'), ignore_index = False, value_name = 'probabilities').reset_index().rename(target_columns, axis = 'columns')
    p_humans_wv = world_vowel_sort(p_humans_wv)
    p_humans_wv.to_csv('probabilities/p_humans_wv.csv')


# french human responses
try:
    p_humansFR_wv = pandas.read_csv('probabilities/p_humansFR_wv.csv')
except FileNotFoundError:
    target_columns = {'language_stimuli': 'language', 'phone': 'vowel', 'filename': 'file'}
    p_humansFR_wv = pandas.melt(pandas.crosstab([fr_human_responses[column] for column in target_columns.keys()], fr_human_responses['assimilation'], colnames = ['classification'], normalize = 'index'), ignore_index = False, value_name = 'probabilities').reset_index().rename(target_columns, axis = 'columns')
    p_humansFR_wv = world_vowel_sort(p_humansFR_wv)
    p_humansFR_wv.to_csv('probabilities/p_humansFR_wv.csv')

# CTC centre probabilities
try:
    p_w2v2_ctc_1_timit_centreMeans_vowels_wv = pandas.read_csv('probabilities/p_w2v2_ctc_1_timit_centreMeans_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2_ctc_1_timit')
    with open('../models/m_w2v2_ctc_1_timit/vocab.json') as f:
        vocab = json.load(f)
    id2label_ctc = {vocab[timit_vowel]: human_vowel for timit_vowel, human_vowel in timit_human_vowels.items()}
    p_w2v2_ctc_1_timit_centreMeans_vowels_wv = probabilities(ctc_wrapper(ctc_model), world_vowels, id2label_ctc)
    p_w2v2_ctc_1_timit_centreMeans_vowels_wv = world_vowel_sort(p_w2v2_ctc_1_timit_centreMeans_vowels_wv)
    p_w2v2_ctc_1_timit_centreMeans_vowels_wv.to_csv('probabilities/p_w2v2_ctc_1_timit_centreMeans_vowels_wv.csv')

# CTC centre probabilities fully frozen
try:
    p_w2v2_ctc_2_timit_centreMeans_vowels_wv = pandas.read_csv('probabilities/p_w2v2_ctc_2_timit_centreMeans_vowels_wv.csv')
except FileNotFoundError:
    ctc_model_frozen = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2_ctc_2_timit')
    with open('../models/m_w2v2_ctc_2_timit/vocab.json') as f:
        vocab = json.load(f)
    id2label_ctc = {vocab[timit_vowel]: human_vowel for timit_vowel, human_vowel in timit_human_vowels.items()}
    p_w2v2_ctc_2_timit_centreMeans_vowels_wv = probabilities(ctc_wrapper(ctc_model_frozen), world_vowels, id2label_ctc)
    p_w2v2_ctc_2_timit_centreMeans_vowels_wv = world_vowel_sort(p_w2v2_ctc_2_timit_centreMeans_vowels_wv)
    p_w2v2_ctc_2_timit_centreMeans_vowels_wv.to_csv('probabilities/p_w2v2_ctc_2_timit_centreMeans_vowels_wv.csv')

# CTC CVC beam search fully frozen
try:
    p_w2v2_ctc_2_timit_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2_ctc_2_timit_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2_ctc_2_timit')
    with open('../models/m_w2v2_ctc_2_timit/vocab.json') as f:
        vocab = json.load(f)
    consonants = ['b', 'ch', 'd', 'dh', 'dx', 'er', 'f', 'g', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh']
    consonant_ids = [vocab[consonant] for consonant in consonants]
    vowel_id2label = {v: timit_human_vowels[k] for k, v in vocab.items() if k in timit_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2_ctc_2_timit_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2_ctc_2_timit_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2_ctc_2_timit_cvcBeamSearch_vowels_wv)
    p_w2v2_ctc_2_timit_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2_ctc_2_timit_cvcBeamSearch_vowels_wv.csv')

# CTC CVC beam search fully frozen v2
try:
    p_w2v2_ctc_2_timit_v2_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2_ctc_2_timit_v2_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2_ctc_2_timit_v2')
    with open('../models/m_w2v2_ctc_2_timit_v2/vocab.json') as f:
        vocab = json.load(f)
    consonants = ['b', 'ch', 'd', 'dh', 'dx', 'er', 'f', 'g', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh']
    consonant_ids = [vocab[consonant] for consonant in consonants]
    vowel_id2label = {v: timit_human_vowels[k] for k, v in vocab.items() if k in timit_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2_ctc_2_timit_v2_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2_ctc_2_timit_v2_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2_ctc_2_timit_v2_cvcBeamSearch_vowels_wv)
    p_w2v2_ctc_2_timit_v2_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2_ctc_2_timit_v2_cvcBeamSearch_vowels_wv.csv')

# CTC CVC beam search
try:
    p_w2v2_ctc_1_timit_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2_ctc_1_timit_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2_ctc_1_timit')
    with open('../models/m_w2v2_ctc_1_timit/vocab.json') as f:
        vocab = json.load(f)
    consonants = ['b', 'ch', 'd', 'dh', 'dx', 'er', 'f', 'g', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh']
    consonant_ids = [vocab[consonant] for consonant in consonants]
    vowel_id2label = {v: timit_human_vowels[k] for k, v in vocab.items() if k in timit_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2_ctc_1_timit_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2_ctc_1_timit_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2_ctc_1_timit_cvcBeamSearch_vowels_wv)
    p_w2v2_ctc_1_timit_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2_ctc_1_timit_cvcBeamSearch_vowels_wv.csv')

# French CTC centre probabilities
try:
    p_w2v2fr_ctc_1_bl_centreMeans_vowels_wv = pandas.read_csv('probabilities/p_w2v2fr_ctc_1_bl_centreMeans_vowels_wv.csv')
except FileNotFoundError:
    fr_ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2fr_ctc_1_bl')
    with open('../models/m_w2v2fr_ctc_1_bl/vocab.json') as f:
        vocabFR = json.load(f)
    fr_id2label_ctc = {vocabFR[bl_vowel]: human_vowel for bl_vowel, human_vowel in bl_human_vowels.items()}
    p_w2v2fr_ctc_1_bl_centreMeans_vowels_wv = probabilities(ctc_wrapper(fr_ctc_model), world_vowels, fr_id2label_ctc)
    p_w2v2fr_ctc_1_bl_centreMeans_vowels_wv = world_vowel_sort(p_w2v2fr_ctc_1_bl_centreMeans_vowels_wv)
    p_w2v2fr_ctc_1_bl_centreMeans_vowels_wv.to_csv('probabilities/p_w2v2fr_ctc_1_bl_centreMeans_vowels_wv.csv')

# French v2 CTC centre probabilities
try:
    p_w2v2fr_ctc_1_bl_v2_centreMeans_vowels_wv = pandas.read_csv('probabilities/p_w2v2fr_ctc_1_bl_v2_centreMeans_vowels_wv.csv')
except FileNotFoundError:
    fr_ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2fr_ctc_1_bl_v2')
    with open('../models/m_w2v2fr_ctc_1_bl_v2/vocab.json') as f:
        vocabFR = json.load(f)
    fr_id2label_ctc = {vocabFR[bl_vowel]: human_vowel for bl_vowel, human_vowel in bl_human_vowels.items()}
    p_w2v2fr_ctc_1_bl_v2_centreMeans_vowels_wv = probabilities(ctc_wrapper(fr_ctc_model), world_vowels, fr_id2label_ctc)
    p_w2v2fr_ctc_1_bl_v2_centreMeans_vowels_wv = world_vowel_sort(p_w2v2fr_ctc_1_bl_v2_centreMeans_vowels_wv)
    p_w2v2fr_ctc_1_bl_v2_centreMeans_vowels_wv.to_csv('probabilities/p_w2v2fr_ctc_1_bl_v2_centreMeans_vowels_wv.csv')

# French CTC CVC beam search
try:
    p_w2v2fr_ctc_1_bl_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2fr_ctc_1_bl_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2fr_ctc_1_bl')
    with open('../models/m_w2v2fr_ctc_1_bl/vocab.json') as f:
        vocab = json.load(f)
    consonants = ['n', 'b', 'k', 's', 'Z', 'v', 'j', 'm', 'w', 'g', 't', 'R', 'l', 'd', 'S', 'N', 'z', 'p', 'f']
    consonant_ids = [vocab[consonant] for consonant in consonants]
    vowel_id2label = {v: bl_human_vowels[k] for k, v in vocab.items() if k in bl_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2fr_ctc_1_bl_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2fr_ctc_1_bl_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2fr_ctc_1_bl_cvcBeamSearch_vowels_wv)
    p_w2v2fr_ctc_1_bl_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2fr_ctc_1_bl_cvcBeamSearch_vowels_wv.csv')

# French BL noisy+substrings CTC CVC beam search
try:
    p_w2v2fr_ctc_1_blNS_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2fr_ctc_1_blNS_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2fr_ctc_1_blNS')
    with open('../models/m_w2v2fr_ctc_1_blNS/vocab.json') as f:
        vocab = json.load(f)
    consonant_ids = [vocab[consonant] for consonant in bl_consonants]
    vowel_id2label = {v: bl_human_vowels[k] for k, v in vocab.items() if k in bl_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2fr_ctc_1_blNS_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2fr_ctc_1_blNS_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2fr_ctc_1_blNS_cvcBeamSearch_vowels_wv)
    p_w2v2fr_ctc_1_blNS_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2fr_ctc_1_blNS_cvcBeamSearch_vowels_wv.csv')

# French v2 CTC CVC beam search
try:
    p_w2v2fr_ctc_1_bl_v2_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2fr_ctc_1_bl_v2_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2fr_ctc_1_bl_v2')
    with open('../models/m_w2v2fr_ctc_1_bl_v2/vocab.json') as f:
        vocab = json.load(f)
    consonant_ids = [vocab[consonant] for consonant in bl_consonants]
    vowel_id2label = {v: bl_human_vowels[k] for k, v in vocab.items() if k in bl_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2fr_ctc_1_bl_v2_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2fr_ctc_1_bl_v2_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2fr_ctc_1_bl_v2_cvcBeamSearch_vowels_wv)
    p_w2v2fr_ctc_1_bl_v2_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2fr_ctc_1_bl_v2_cvcBeamSearch_vowels_wv.csv')

# French Librispeech CTC CVC beam search
try:
    p_w2v2fr_ctc_1_librispeechFR_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2fr_ctc_1_librispeechFR_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2fr_ctc_1_librispeechFR')
    with open('../models/m_w2v2fr_ctc_1_librispeechFR/vocab.json', encoding='utf-8') as f:
        vocab = json.load(f)
    consonants = ['b', 'd', 'dʒ', 'f', 'j', 'k', 'l', 'm', 'n', 'p', 's', 't', 'tʃ', 'v', 'w', 'z', 'ɡ', 'ɲ', 'ʁ', 'ʃ', 'ʒ']
    consonant_ids = [vocab[consonant] for consonant in consonants]
    vowel_id2label = {v: k for k, v in vocab.items() if k in human_bl_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2fr_ctc_1_librispeechFR_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2fr_ctc_1_librispeechFR_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2fr_ctc_1_librispeechFR_cvcBeamSearch_vowels_wv)
    p_w2v2fr_ctc_1_librispeechFR_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2fr_ctc_1_librispeechFR_cvcBeamSearch_vowels_wv.csv')

# Transformer CTC beam search
try:
    p_w2v2_transformer_ctc_2_timit_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2_transformer_ctc_2_timit_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTCWithTransformer.from_pretrained('../models/m_w2v2_transformer_ctc_2_timit')
    with open('../models/m_w2v2_transformer_ctc_2_timit/vocab.json') as f:
        vocab = json.load(f)
    consonants = ['b', 'ch', 'd', 'dh', 'dx', 'er', 'f', 'g', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh']
    consonant_ids = [vocab[consonant] for consonant in consonants]
    vowel_id2label = {v: timit_human_vowels[k] for k, v in vocab.items() if k in timit_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2_transformer_ctc_2_timit_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2_transformer_ctc_2_timit_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2_transformer_ctc_2_timit_cvcBeamSearch_vowels_wv)
    p_w2v2_transformer_ctc_2_timit_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2_transformer_ctc_2_timit_cvcBeamSearch_vowels_wv.csv')

    
# Transformer CTC beam search v2
try:
    p_w2v2_transformer_ctc_2_timit_v2_cvcBeamSearch_vowels_wv = pandas.read_csv('probabilities/p_w2v2_transformer_ctc_2_timit_v2_cvcBeamSearch_vowels_wv.csv')
except FileNotFoundError:
    ctc_model = Wav2Vec2ForCTCWithTransformer.from_pretrained('../models/m_w2v2_transformer_ctc_2_timit_v2')
    with open('../models/m_w2v2_transformer_ctc_2_timit_v2/vocab.json') as f:
        vocab = json.load(f)
    consonants = ['b', 'ch', 'd', 'dh', 'dx', 'er', 'f', 'g', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh']
    consonant_ids = [vocab[consonant] for consonant in consonants]
    vowel_id2label = {v: timit_human_vowels[k] for k, v in vocab.items() if k in timit_human_vowels.keys()}
    padding_token_id = vocab['<pad>']
    beam_width = 100
    p_w2v2_transformer_ctc_2_timit_v2_cvcBeamSearch_vowels_wv = probabilities(
        ctc_cvcBeamSearch_wrapper(ctc_model, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width),
        world_vowels
    )
    p_w2v2_transformer_ctc_2_timit_v2_cvcBeamSearch_vowels_wv = world_vowel_sort(p_w2v2_transformer_ctc_2_timit_v2_cvcBeamSearch_vowels_wv)
    p_w2v2_transformer_ctc_2_timit_v2_cvcBeamSearch_vowels_wv.to_csv('probabilities/p_w2v2_transformer_ctc_2_timit_v2_cvcBeamSearch_vowels_wv.csv')

# prev_phone = human_responses.set_index('filename')['prev_phone'].groupby('filename').first().rename_axis(index = 'file')
# next_phone = human_responses.set_index('filename')['next_phone'].groupby('filename').first().rename_axis(index = 'file')

# world_vowel_probabilities.insert(len(world_vowel_probabilities.columns), 'prev_phone', prev_phone.loc[world_vowel_probabilities['file']].reset_index()['prev_phone'])
# human_responses_pooled.insert(len(human_responses_pooled.columns), 'prev_phone', prev_phone.loc[human_responses_pooled['file']].reset_index()['prev_phone'])
# world_vowels_ctc.insert(len(world_vowels_ctc.columns), 'prev_phone', prev_phone.loc[world_vowels_ctc['file']].reset_index()['prev_phone'])
# world_vowel_probabilities.insert(len(world_vowel_probabilities.columns), 'next_phone', next_phone.loc[world_vowel_probabilities['file']].reset_index()['next_phone'])
# human_responses_pooled.insert(len(human_responses_pooled.columns), 'next_phone', next_phone.loc[human_responses_pooled['file']].reset_index()['next_phone'])
# world_vowels_ctc.insert(len(world_vowels_ctc.columns), 'next_phone', next_phone.loc[world_vowels_ctc['file']].reset_index()['next_phone'])


# formants = pandas.read_csv('../stimuli_world_vowels/formants/world_vowels_formants.csv')
# formants = formants.set_index('file')
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