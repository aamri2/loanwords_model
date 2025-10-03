from datasets import load_from_disk
import json
from collections import defaultdict, namedtuple
from model_handler import pool
import seaborn
import matplotlib.pyplot as plt
import numpy as np

TrigramProbabilities = namedtuple('TrigramProbabilities', ['unigrams', 'CV_bigrams', 'VC_bigrams', 'CVC_trigrams'])

def get_trigram_probabilities(dataset, label2id, vowels, consonants, as_strings = True):
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
            unigrams[char] += 1
            if char in vowels:
                if i > 0 and text[i - 1] in consonants:
                    CV_bigrams[char][text[i - 1]] += 1
                    CV = True
                if i < len(text) - 1 and text[i + 1] in consonants:
                    VC_bigrams[char][text[i + 1]] += 1
                    VC = True
                if CV and VC:
                    CVC_trigrams[char][(text[i - 1], text[i + 1])] += 1
    
    # translate to characters
    unigrams = {id2label[k]: v for k, v in unigrams.items()}
    CV_bigrams = {id2label[V]: {id2label[C]: freqs for C, freqs in C_freqs.items()} for V, C_freqs in CV_bigrams.items()}
    VC_bigrams = {id2label[V]: {id2label[C]: freqs for C, freqs in C_freqs.items()} for V, C_freqs in VC_bigrams.items()}
    CVC_trigrams = {id2label[V]: {tuple(id2label[C] for C in CC): freqs for CC, freqs in CC_freqs.items()} for V, CC_freqs in CVC_trigrams.items()}

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