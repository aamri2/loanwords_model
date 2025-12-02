from datasets import load_from_disk
import json
from collections import defaultdict, namedtuple
from model_handler import pool
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TrigramFrequencies = namedtuple('TrigramFrequencies', ['unigrams', 'CV_bigrams', 'VC_bigrams', 'CVC_trigrams'])
TrigramProbabilities = namedtuple('TrigramProbabilities', ['unigrams', 'CV_bigrams', 'VC_bigrams', 'CVC_trigrams'])

def get_trigram_frequencies(dataset, label2id, vowels, consonants, as_strings = True, word_boundaries = False):
    """If word_boundaries is true, only get word-initial CVs and word-final VCs. Assumes '|' as word boundary."""

    if as_strings:
        vowels = [label2id[vowel] for vowel in vowels]
        consonants = [label2id[consonant] for consonant in consonants]
    
    id2label = {v: k for k, v in label2id.items()}
    # get conditional frequencies
    all_text: list[list[int]] = sum([dataset[split]['labels'] for split in dataset.keys()], [])
    unigrams = defaultdict(lambda: 0)
    CV_bigrams = defaultdict(lambda: defaultdict(lambda: 0)) # C1 given V; [V][C]
    VC_bigrams = defaultdict(lambda: defaultdict(lambda: 0)) # C2 given V; [V][C]
    CVC_trigrams = defaultdict(lambda: defaultdict(lambda: 0)) # C1_C2 given V; [V][C]
    for text in all_text:
        for i, char in enumerate(text):
            CV, VC = False, False
            if not char == '|':
                unigrams[char] += 1
            if char == '|':
                if i > 0 and text[i] in consonants and text[i + 1] in vowels:
                    CV_bigrams[text[i + 1]][text[i]] += 1
                    CV = True
                if i < len(text) and text[i - 2] in consonants and text[i - 1] in vowels:
                    VC_bigrams[text[i + 1]][text[i + 1]] += 1
                    VC = True
                if CV and VC:
                    CVC_trigrams[char][(text[i - 1], text[i + 1])] += 1
    
    # translate to characters
    unigrams = {id2label[k]: v for k, v in unigrams.items()}
    CV_bigrams = {id2label[V]: {id2label[C]: freqs for C, freqs in C_freqs.items()} for V, C_freqs in CV_bigrams.items()}
    VC_bigrams = {id2label[V]: {id2label[C]: freqs for C, freqs in C_freqs.items()} for V, C_freqs in VC_bigrams.items()}
    CVC_trigrams = {id2label[V]: {tuple(id2label[C] for C in CC): freqs for CC, freqs in CC_freqs.items()} for V, CC_freqs in CVC_trigrams.items()}

    return TrigramFrequencies(unigrams=unigrams, CV_bigrams=CV_bigrams, VC_bigrams=VC_bigrams, CVC_trigrams=CVC_trigrams)
    
def get_trigram_probabilities(dataset, label2id, vowel, consonants, as_strings = True, word_boundaries = False):
    unigrams, CV_bigrams, VC_bigrams, CVC_trigrams = get_trigram_frequencies(dataset, label2id, vowel, consonants, as_strings, word_boundaries)
    # convert to probabilities
    n_unigrams = sum(v for k, v in unigrams.items())
    for k in unigrams.keys():
        unigrams[k] /= n_unigrams
    for V, C_freqs in CV_bigrams.items():
        n_V = sum(freqs for C, freqs in C_freqs.items())
        for C in C_freqs.keys():
            CV_bigrams[V][C] /= n_V
    for V, C_freqs in VC_bigrams.items():
        n_V = sum(freqs for C, freqs in C_freqs.items())
        for C in C_freqs.keys():
            VC_bigrams[V][C] /= n_V
    for V, CC_freqs in CVC_trigrams.items():
        n_V = sum(freqs for CC, freqs in CC_freqs.items())
        for CC in CC_freqs.keys():
            CVC_trigrams[V][CC] /= n_V
    
    return TrigramProbabilities(unigrams=unigrams, CV_bigrams=CV_bigrams, VC_bigrams=VC_bigrams, CVC_trigrams=CVC_trigrams)

librispeechCL = load_from_disk('../prep_librispeechCL')
with open('../prep_librispeechCL/vocab.json') as f:
    vocab = json.load(f)

librispeech_vowels = ['i', 'ɪ', 'eɪ', 'ɛ', 'æ', 'ɑ', 'ʌ', 'oʊ', 'u', 'ʊ']
librispeech_consonants = ["b", "d", "dʒ", "f", "h", "j", "k", "l", "m", "n", "p", "s", "t", "tʃ", "v", "w", "z", "ð", "ŋ", "ɡ", "ɹ", "ɾ", "ʃ", "ʒ", "θ"]

trigram_frequencies = get_trigram_frequencies(librispeechCL, vocab, librispeech_vowels, librispeech_consonants)

trigram_probabilities = get_trigram_probabilities(librispeechCL, vocab, librispeech_vowels, librispeech_consonants)

def get_error_rates(probabilities, **kwargs):
    """Expects the classifications to correspond to the vowels."""

    p = pool(probabilities, 'vowel', **kwargs)

def get_correct_vowel(row):
    vowel = row.name[1]
    return row[vowel]

def bigram_freq_scatter(probabilities, trigram_probabilities: TrigramProbabilities, **kwargs):
    if isinstance(probabilities, list):
        for probability in probabilities:
            p = pool(probability, 'prev_phone', 'vowel', 'next_phone', **kwargs)
            p['bigram_freq'] = [trigram_probabilities.CV_bigrams[V].get(C1, 1e-16)*trigram_probabilities.VC_bigrams[V].get(C2, 1e-16)*trigram_probabilities.unigrams[V] for C1, V, C2 in p.index]
            p['prob'] = p.apply(get_correct_vowel, axis=1)
            p['prob'] = np.log(p['prob'])
            p['bigram_freq'] = np.log(p['bigram_freq'])
            p['label'] = [i + j + k for i, j, k in zip(p.reset_index()['prev_phone'], p.reset_index()['vowel'], p.reset_index()['next_phone'])]
            seaborn.scatterplot(p.reset_index(), x='bigram_freq', y='prob', hue='label').set_ylim(ymin=-5, ymax=0)
            plt.figure()
        plt.show()
    else:
        p = pool(probabilities, 'prev_phone', 'vowel', 'next_phone', **kwargs)
        p['bigram_freq'] = [trigram_probabilities.CV_bigrams[V].get(C1, 1e-16)*trigram_probabilities.VC_bigrams[V].get(C2, 1e-16)*trigram_probabilities.unigrams[V] for C1, V, C2 in p.index]
        p['prob'] = p.apply(get_correct_vowel, axis=1)
        p['prob'] = np.log(p['prob'])
        p['bigram_freq'] = np.log(p['bigram_freq'])
        p['label'] = [i + j + k for i, j, k in zip(p.reset_index()['prev_phone'], p.reset_index()['vowel'], p.reset_index()['next_phone'])]
        seaborn.scatterplot(p.reset_index(), x='bigram_freq', y='prob', hue='label').set_ylim(ymin=-5, ymax=0)
        plt.show()


def trigram_freq_scatter(probabilities, trigram_probabilities: TrigramProbabilities, **kwargs):
    p = pool(probabilities, 'prev_phone', 'vowel', 'next_phone', **kwargs)
    p['trigram_freq'] = [trigram_probabilities.CVC_trigrams[V].get((C1, C2), 1e-16)*trigram_probabilities.unigrams[V] for C1, V, C2 in p.index]
    p['prob'] = p.apply(get_correct_vowel, axis=1)
    p['prob'] = np.log(p['prob'])
    p['trigram_freq'] = np.log(p['trigram_freq'])
    seaborn.scatterplot(p, x='trigram_freq', y='prob', hue='vowel')
    plt.show()

def plot_file_and_freqs(probabilities, file, trigram_probabilities: TrigramProbabilities, **kwargs):
    p = pool(probabilities, 'prev_phone', 'vowel', 'next_phone', file = file, **kwargs)
    prev_phone = p.index.item()[0]
    next_phone = p.index.item()[1]
    vowel = p.index.item()[2]
    p = p.reset_index(drop=True)
    cvc_freqs = {col: [trigram_probabilities.CV_bigrams[col][c1]*trigram_probabilities.VC_bigrams[col][c2] for c1 in trigram_probabilities.CV_bigrams[col] for c2 in trigram_probabilities.VC_bigrams[col]] for col in p.columns}
    v_freqs = {col: trigram_probabilities.unigrams[col] for col in p.columns}
    cvc_freq = {col: trigram_probabilities.CV_bigrams[col][prev_phone]*trigram_probabilities.VC_bigrams[col][next_phone] for col in p.columns}
    
    p.reset_index(level=['prev_phone', 'next_phone'])
    bigram_probabilities = trigram_probabilities.CV_bigrams[V][C] 
    pd.DataFrame({column: [row[column], trigram_probabilities] for column in columns})

# TODO
correct_p = p.loc[(p['prev_phone'].apply(lambda x: wv_timit_consonants.get(x, x)) == p['onset_classification']) * (p['vowel'] == p['vowel_classification']) * (p['next_phone'].apply(lambda x: wv_timit_consonants.get(x, x)) == p['coda_classification'])][['language', 'vowel', 'file', 'probabilities', 'prev_phone', 'next_phone']]
correct_p = correct_p.set_index('file')
correct_p = correct_p.rename({'probabilities': 'correct_probabilities'}, axis = 1)
selected_p = p.pivot_table(values = 'probabilities', columns = ['onset_classification', 'vowel_classification', 'coda_classification'], index = 'file', sort = False).idxmax(1)
selected_p = pandas.DataFrame(selected_p.to_list(), index=selected_p.index, columns = ['onset_classification', 'vowel_classification', 'coda_classification'])
selected_p['selected_probabilities'] = p.set_index(['file', 'onset_classification', 'vowel_classification', 'coda_classification']).loc[zip(selected_p.index, selected_p['onset_classification'], selected_p['vowel_classification'], selected_p['coda_classification'])].reset_index().set_index('file')['probabilities']
p = pandas.merge(correct_p, selected_p, left_index = True, right_index = True)
p['intended'] = p['prev_phone'] + p['vowel'] + p['next_phone']
p['selected'] = p['onset_classification'] + p['vowel_classification'] + p['coda_classification']

timit_librispeech_consonants = {'jh': 'dʒ', 'dh': 'ð', 'g': 'ɡ', 'sh': 'ʃ', 'th': 'θ', 'ch': 'tʃ', 'ng': 'ŋ'}
wv_librispeech_consonants = {'r': 'ɹ', 'g': 'ɡ'}
epsilon = 1e-2
p['selected_textual_probabilities'] = p.apply(lambda x: numpy.exp(numpy.log([trigram_probabilities.unigrams[x.vowel_classification],
                        trigram_probabilities.CV_bigrams[x.vowel_classification].get(timit_librispeech_consonants.get(x.onset_classification, x.onset_classification), epsilon),
                        trigram_probabilities.VC_bigrams[x.vowel_classification].get(timit_librispeech_consonants.get(x.coda_classification, x.coda_classification), epsilon)]).sum()), axis = 1)
p['correct_textual_probabilities'] = p.apply(lambda x: numpy.exp(numpy.log([trigram_probabilities.unigrams[x.vowel],
                        trigram_probabilities.CV_bigrams[x.vowel][wv_librispeech_consonants.get(x.prev_phone, x.prev_phone)],
                        trigram_probabilities.VC_bigrams[x.vowel][wv_librispeech_consonants.get(x.next_phone, x.next_phone)]]).sum()), axis = 1)

p['log_textual_probabilities_ratio'] = p.apply(lambda x: numpy.log(x['selected_textual_probabilities']) - numpy.log(x['correct_textual_probabilities']), axis = 1)
p['log_probabilities_ratio'] = p.apply(lambda x: numpy.log(x['selected_probabilities']) - numpy.log(x['correct_probabilities']), axis = 1)
p['textual_probabilities_ratio'] = p.apply(lambda x: numpy.exp(x['log_textual_probabilities_ratio']), axis = 1)
p['probabilities_ratio'] = p.apply(lambda x: numpy.exp(x['log_probabilities_ratio']), axis = 1)

p['log_probabilities_ratios_ratio'] = p.apply(lambda x: x['log_textual_probabilities_ratio'] - x['log_probabilities_ratio'], axis = 1)
p['probabilities_ratios_ratio'] = p.apply(lambda x: numpy.exp(x['log_probabilities_ratios_ratio']), axis = 1)

seaborn.histplot(p[p['lang'] == 'EN'], x = 'log_textual_probabilities_ratio'); plt.show()
seaborn.histplot(p, x = 'log_probabilities_ratio'); plt.show()
seaborn.histplot(p, x = 'log_probabilities_ratios_ratio'); plt.show()
numpy.exp(p['log_textual_probabilities_ratio'].mean())
seaborn.regplot(p, y = 'log_probabilities_ratio', x = 'log_textual_probabilities_ratio'); plt.show()
seaborn.regplot(p, x = 'log_')

# timit instead
librispeechCL = load_from_disk('../prep_librispeechCL')
with open('../prep_librispeechCL/vocab.json') as f:
    vocab = json.load(f)

librispeech_vowels = ['i', 'ɪ', 'eɪ', 'ɛ', 'æ', 'ɑ', 'ʌ', 'oʊ', 'u', 'ʊ']
librispeech_consonants = ["b", "d", "dʒ", "f", "h", "j", "k", "l", "m", "n", "p", "s", "t", "tʃ", "v", "w", "z", "ð", "ŋ", "ɡ", "ɹ", "ɾ", "ʃ", "ʒ", "θ"]
trigram_frequencies = get_trigram_frequencies(librispeechCL, vocab, librispeech_vowels, librispeech_consonants)
trigram_probabilities = get_trigram_probabilities(librispeechCL, vocab, librispeech_vowels, librispeech_consonants)

# librispeech word boundaries only
from dataset_handler import prepare_librispeech_ctc
librispeechCLwords, vocab = prepare_librispeech_ctc(True, True)
