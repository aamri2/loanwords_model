from datasets import load_from_disk
import json
from collections import defaultdict
from model_handler import pool
import seaborn
import matplotlib.pyplot as plt
import numpy as np

librispeechCL = load_from_disk('../prep_librispeechCL')
with open('../prep_librispeechCL/vocab.json') as f:
    vocab = json.load(f)

reverse_vocab = {v: k for k, v in vocab.items()}

librispeech_vowels = ['i', 'ɪ', 'eɪ', 'ɛ', 'æ', 'ɑ', 'ʌ', 'oʊ', 'u', 'ʊ']
vowels = [vocab[vowel] for vowel in librispeech_vowels]

# get (conditional) frequencies
all_text: list[list[int]] = librispeechCL['train.clean.100']['labels'] + librispeechCL['validation.clean']['labels'] # type:ignore # known to be DatasetDict
unigrams = defaultdict(lambda: 0)
CV_bigrams = defaultdict(lambda: defaultdict(lambda: 0)) # C1 given V; [V][C]
VC_bigrams = defaultdict(lambda: defaultdict(lambda: 0)) # C2 given V; [V][C]
CVC_trigrams = defaultdict(lambda: defaultdict(lambda: 0)) # C1_C2 given V; [V][C]
for text in all_text:
    for i, char in enumerate(text):
        CV, VC = False, False
        unigrams[char] += 1
        if char in vowels:
            if i > 0 and text[i - 1] not in vowels:
                CV_bigrams[char][text[i - 1]] += 1
                CV = True
            if i < len(text) - 1 and text[i + 1] not in vowels:
                VC_bigrams[char][text[i + 1]] += 1
                VC = True
            if CV and VC:
                CVC_trigrams[char][(text[i - 1], text[i + 1])] += 1

# translate to characters
unigrams = {reverse_vocab[k]: v for k, v in unigrams.items()}
CV_bigrams = {reverse_vocab[V]: {reverse_vocab[C]: freqs for C, freqs in C_freqs.items()} for V, C_freqs in CV_bigrams.items()}
VC_bigrams = {reverse_vocab[V]: {reverse_vocab[C]: freqs for C, freqs in C_freqs.items()} for V, C_freqs in VC_bigrams.items()}
CVC_trigrams = {reverse_vocab[V]: {tuple(reverse_vocab[C] for C in CC): freqs for CC, freqs in CC_freqs.items()} for V, CC_freqs in CVC_trigrams.items()}

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

def get_correct_vowel(row):
    vowel = row.name[1]
    return row[vowel]

def bigram_freq_scatter(probabilities, **kwargs):
    p = pool(probabilities, 'prev_phone', 'vowel', 'next_phone', **kwargs)
    p['bigram_freq'] = [CV_bigrams[V].get(C1, 1e-16)*VC_bigrams[V].get(C2, 1e-16)*unigrams[V] for C1, V, C2 in p.index]
    p['prob'] = p.apply(get_correct_vowel, axis=1)
    p['prob'] = np.log(p['prob'])
    p['bigram_freq'] = np.log(p['bigram_freq'])
    seaborn.scatterplot(p, x='bigram_freq', y='prob')
    plt.show()


def trigram_freq_scatter(probabilities, **kwargs):
    p = pool(probabilities, 'prev_phone', 'vowel', 'next_phone', **kwargs)
    p['trigram_freq'] = [CVC_trigrams[V].get((C1, C2), 1e-16)*unigrams[V] for C1, V, C2 in p.index]
    p['prob'] = p.apply(get_correct_vowel, axis=1)
    p['prob'] = np.log(p['prob'])
    p['trigram_freq'] = np.log(p['trigram_freq'])
    seaborn.scatterplot(p, x='trigram_freq', y='prob')
    plt.show()