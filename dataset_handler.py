from spec import TrainingDatasetSpec, _SEPARATOR
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_from_disk
import datasets
import json
import pandas as pd
from typing import Union, cast
from collections.abc import Callable
import phonemizer.separator
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PhonemeCTCTokenizer
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
import phonemizer
from pyacoustics.speech_filters import speech_shaped_noise
import numpy as np
from torchaudio.functional import forced_align, merge_tokens

_DATASET_PATH = '../'
_DATASET_PREFIX = 'prep'

class TrainingDataset():
    """
    Uses TrainingDatasetSpec to parse and load datasets.
    """

    spec: TrainingDatasetSpec
    path: str
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict
    vocab: dict[str, int]
    consonants: list[str]
    vowels: list[str]

    def __init__(self, spec: str | TrainingDatasetSpec, path: str | None = None):
        self.spec = TrainingDatasetSpec(spec)
        self.path = path if path else f'{_DATASET_PATH}{_DATASET_PREFIX}{_SEPARATOR}{spec}'
        self.dataset = self.load_dataset()
        self.vocab = self.get_dataset_vocab()
        self.consonants = get_consonants(self.spec)
        self.vowels = get_vowels(self.spec)
    
    @property
    def vocab_path(self) -> str:
        return self.path + '/vocab.json'

    def load_dataset(self) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
        """Loads a dataset given its specification."""
        
        return datasets.load_from_disk(self.path)


    def get_dataset_vocab(self) -> dict[str, int]:
        """Loads a dataset's vocabulary given its specification."""

        with open(self.vocab_path, encoding = 'utf-8') as f:
            vocab = json.load(f)
        return vocab

consonants = {
    'timit': ['b', 'ch', 'd', 'dh', 'dx', 'er', 'f', 'g', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh'],
    'librispeech': ["b", "d", "dʒ", "f", "h", "j", "k", "l", "m", "n", "p", "s", "t", "tʃ", "v", "w", "z", "ð", "ŋ", "ɡ", "ɹ", "ɾ", "ʃ", "ʒ", "θ"],
    'librispeechFR': ['b', 'd', 'dʒ', 'f', 'j', 'k', 'l', 'm', 'n', 'p', 's', 't', 'tʃ', 'v', 'w', 'z', 'ɡ', 'ɲ', 'ʁ', 'ʃ', 'ʒ'],
    'bl': ['n', 'b', 'k', 's', 'Z', 'v', 'j', 'm', 'w', 'g', 't', 'R', 'l', 'd', 'S', 'N', 'z', 'p', 'f']
}

vowels = {
    
}

def get_consonants(spec: str | TrainingDatasetSpec) -> list[str]:
    """Given a specification, gets the consonants in the transcription system of the dataset."""

    spec = TrainingDatasetSpec(spec)
    return consonants[spec.family]

def get_vowels(spec: str | TrainingDatasetSpec) -> list[str]:
    """Given a specification, gets the vowels in the transcription system of the dataset."""

    spec = TrainingDatasetSpec(spec)
    return vowels[spec.family]

def make_noisy(dataset, snr):
    """Takes a prepared dataset and returns it with noise added."""

    def add_noise(batch):
        audios = [np.array(input_values) for input_values in batch['input_values']]
        all_audio = np.hstack(audios)
        noise = speech_shaped_noise._noise_from_signal(all_audio)
        for i, audio in enumerate(audios):
            level = speech_shaped_noise._dbspl(audio)
            audios[i] = speech_shaped_noise._mix_noise(audio, noise, level, snr)[1].tolist()
        return {'input_values': audios}
    
    dataset = dataset.map(add_noise, batched=True, batch_size=64)
    return dataset

def align_ctc(dataset, model):
    """
    Given a prepared dataset and a CTC model trained on that dataset, gives alignments.
    
    Alignments are given in the columns "start" and "end" of the returned dataset.
    """

    def get_alignments(batch):
        input_values = torch.tensor([batch['input_values']])
        with torch.no_grad():
            log_probs = model(input_values).logits
        targets = torch.tensor([batch['labels']])
        tokens, scores = forced_align(log_probs, targets, blank=model.config.pad_token_id)
        merged_tokens = merge_tokens(tokens[0], scores[0], blank=model.config.pad_token_id)
        start, end = zip(*[(token_span.start, token_span.end) for token_span in merged_tokens])
        batch['start'] = start
        batch['end'] = end
        return batch

    dataset = dataset.map(get_alignments)
    return dataset

def random_substrings(aligned_dataset, samples_to_frames_ratio, substring_length = None, substring_ratio = 0.3):
    """
    Takes a prepared dataset with alignments and returns randomly selected substrings.

    Args:
        aligned_dataset: A prepared dataset with additional
            columns 'start' and 'end' containing frame-alignments.
        samples_to_frames_ratio: The number of samples per frame, where alignment
            is at the frame-level.
        substring_length: Takes precedence over 'substring_ratio'. The length
            of the desired substrings, expressed in number of tokens.
        substring_ratio: The length of the desired substrings, expressed
            as a ratio of the total number of tokens.
    """

    def use_substring_length(string_length):
        return torch.tensor([substring_length]).int()

    def use_substring_ratio(string_length):
        rounding = torch.randint(0, 2, (1,))
        return torch.tensor([string_length * substring_ratio]).int() + rounding
    
    get_substring_length = use_substring_length if substring_length else use_substring_ratio

    def get_substrings(batch):
        n_tokens = get_substring_length(len(batch['labels']))
        start_token = torch.randint(1, len(batch['labels']) - n_tokens, (1,))
        end_token = start_token + n_tokens
        # substring is taken from end of previous token to start of following token
        start_frame = batch['end'][start_token - 1]
        end_frame = batch['start'][end_token]
        audio_substring = batch['input_values'][(np.floor(start_frame * samples_to_frames_ratio - 1).astype('int')):(np.ceil(end_frame * samples_to_frames_ratio).astype('int'))]
        labels_substring = batch['labels'][start_token:end_token]
        batch['input_values'] = audio_substring
        batch['labels'] = labels_substring
        return batch
    
    aligned_dataset = aligned_dataset.map(get_substrings, remove_columns = ['start', 'end'])
    return aligned_dataset

def noisy_and_substrings(aligned_dataset, snr, samples_to_frames_ratio, substring_length = None, substring_ratio = 0.3, noisy_ratio = 0.5, substring_subset_ratio = 0.5, keep_originals = False):
    """Helper function makes noisy and substring split of aligned dataset."""

    if isinstance(aligned_dataset, DatasetDict):
        new_dataset = DatasetDict()
        for key, dataset in aligned_dataset.items():
            new_dataset[key] = noisy_and_substrings(dataset, snr, samples_to_frames_ratio, substring_length, substring_ratio, noisy_ratio, substring_subset_ratio, keep_originals)
        return new_dataset
    
    random_indices = np.random.choice(range(len(aligned_dataset)), size=(len(aligned_dataset,)), replace=False)

    # make noisy subset
    if noisy_ratio > 0:
        noisy_dataset = make_noisy(
            aligned_dataset.select(random_indices[:int(noisy_ratio*len(aligned_dataset))]), snr
        )
    else:
        noisy_dataset = Dataset.from_dict({'input_values': [], 'labels': [], 'start': [], 'end': []})
    if keep_originals:
        clean_dataset = aligned_dataset
    else:
        clean_dataset = aligned_dataset.select(random_indices[int(noisy_ratio*len(aligned_dataset)):])

    # make substring subsets
    if substring_subset_ratio > 0:
        noisy_substring_dataset = random_substrings(
            noisy_dataset.select(range(int(substring_subset_ratio*len(noisy_dataset)))), samples_to_frames_ratio, substring_length, substring_ratio
        )
        clean_substring_dataset = random_substrings(
            clean_dataset.select(range(int(substring_subset_ratio*len(clean_dataset)))), samples_to_frames_ratio, substring_length, substring_ratio
        )
    else:
        noisy_substring_dataset = Dataset.from_dict({'input_values': [], 'labels': []})
        clean_substring_dataset = Dataset.from_dict({'input_values': [], 'labels': []})
    if keep_originals:
        noisy_string_dataset = noisy_dataset.remove_columns(['start', 'end'])
        clean_string_dataset = clean_dataset.remove_columns(['start', 'end'])
    else:
        noisy_string_dataset = noisy_dataset.select(range(int(substring_subset_ratio*len(noisy_dataset)), len(noisy_dataset))).remove_columns(['start', 'end'])
        clean_string_dataset = clean_dataset.select(range(int(substring_subset_ratio*len(clean_dataset)), len(clean_dataset))).remove_columns(['start', 'end'])

    new_dataset = datasets.concatenate_datasets([noisy_substring_dataset, noisy_string_dataset, clean_substring_dataset, clean_string_dataset])
    new_dataset = new_dataset.shuffle().flatten_indices()
    return new_dataset

def throughPretrainedModel(dataset, model):
    """Runs a prepared dataset through a given base model."""

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def process_row(batch):
        input_values = feature_extractor(batch['input_values'], sampling_rate=16000, return_tensors='pt', padding=True)['input_values']
        with torch.no_grad():
            batch['input_values'] = model(input_values).last_hidden_state.detach()
        return batch
        
    return dataset.map(process_row, batched=True, batch_size=8)

def prepare_timitMV_w2v2():
    """Runs TIMIT MV through w2v2."""

    try:
        timit = load_from_disk('../prep_timitMV')
    except FileNotFoundError:
        timit = prepare_masked_targets()
        timit.save_to_disk('../prep_timitMV')

    model = Wav2Vec2Model.from_pretrained('../w2v2')
    
    return throughPretrainedModel(timit, model)

def prepare_wvResponses():
    """Prepares World Vowels stimuli with individual responses as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[human_responses['language_indiv'] == 'english']

    wvResponses = Dataset.from_dict({
        'audio': [f'../stimuli_world_vowels/{file}.wav' for file in human_responses['filename']],
        'label': human_responses['assimilation'],
        'vowel_language': [f'{vowel}_{language}' for vowel, language in zip(human_responses['#phone'], human_responses['language_stimuli'])],
    }).cast_column('audio', datasets.Audio()).class_encode_column('label').class_encode_column('vowel_language')
    
    wvResponses_split = wvResponses.train_test_split(0.2, 0.8, stratify_by_column = 'vowel_language')
    wvResponses_test_dev = wvResponses_split['test'].train_test_split(0.5, 0.5, stratify_by_column = 'vowel_language')
    wvResponses = DatasetDict({'train': wvResponses_split['train'], 'test': wvResponses_test_dev['train'], 'dev': wvResponses_test_dev['test']})

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    prep_wvResponses = wvResponses.map(prepare_dataset, remove_columns=['audio', 'vowel_language'])
    return prep_wvResponses

def prepare_wvENResponses():
    """Prepares World Vowels stimuli with individual responses as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'english') * (human_responses['language_stimuli'] == 'EN')]

    wvENResponses = Dataset.from_dict({
        'audio': [f'../stimuli_world_vowels/{file}.wav' for file in human_responses['filename']],
        'label': human_responses['assimilation'],
        'vowel': human_responses['#phone'],
    }).cast_column('audio', datasets.Audio()).class_encode_column('label').class_encode_column('vowel').shuffle()
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    prep_wvENResponses = wvENResponses.map(prepare_dataset, remove_columns=['audio', 'vowel'])
    return prep_wvENResponses

def prepare_wvENResponses10Fold():
    """Prepares World Vowels stimuli with individual responses as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'english') * (human_responses['language_stimuli'] == 'EN')]

    wvENResponses = Dataset.from_dict({
        'audio': [f'../stimuli_world_vowels/{file}.wav' for file in human_responses['filename']],
        'label': human_responses['assimilation'],
        'vowel': human_responses['#phone'],
    }).cast_column('audio', datasets.Audio()).class_encode_column('label').class_encode_column('vowel').shuffle()
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    prep_wvENResponses = wvENResponses.map(prepare_dataset, remove_columns=['audio'])
    splits = prep_wvENResponses.train_test_split(test_size=1/10)
    prep_wvENResponses_folds = [splits['test']]
    
    for i in reversed(range(2, 10)):
        splits = splits['train'].train_test_split(test_size=1/i)
        prep_wvENResponses_folds.append(splits['test'])
    prep_wvENResponses_folds.append(splits['train'])
    return DatasetDict({f'fold_{i}': prep_wvENResponses_folds[i] for i in range(10)})

def prepare_wvFRResponses10Fold():
    """Prepares French World Vowels stimuli with individual responses (from French-speaking participants) as classifications."""
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'french') * (human_responses['language_stimuli'] == 'FR')]

    wvFRResponses = Dataset.from_dict({
        'audio': [f'../stimuli_world_vowels/{file}.wav' for file in human_responses['filename']],
        'label': human_responses['assimilation'],
        'vowel': human_responses['#phone'],
    }).cast_column('audio', datasets.Audio()).class_encode_column('label').class_encode_column('vowel').shuffle()
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    prep_wvFRResponses = wvFRResponses.map(prepare_dataset, remove_columns=['audio'])
    splits = prep_wvFRResponses.train_test_split(test_size=1/10)
    prep_wvFRResponses_folds = [splits['test']]
    
    for i in reversed(range(2, 10)):
        splits = splits['train'].train_test_split(test_size=1/i)
        prep_wvFRResponses_folds.append(splits['test'])
    prep_wvFRResponses_folds.append(splits['train'])
    return DatasetDict({f'fold_{i}': prep_wvFRResponses_folds[i] for i in range(10)})

def prepare_wvEN():
    """Prepares World Vowels English stimuli."""
    
    wv = pd.read_csv('WorldVowels_stimuli.csv')
    files = ['../wv/' + i for i in wv[wv['language'] == 'EN']['#file_extract']]
    phone = list(wv[wv['language'] == 'EN']['#phone'])
    wvEN = Dataset.from_dict({'audio': files, 'labels': phone}, split=datasets.Split.TRAIN)\
        .cast_column('audio', datasets.Audio())\
        .class_encode_column('labels')
    #wvEN = wvEN.train_test_split(seed=2025, stratify_by_column='labels')

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    prep_wvEN = wvEN.map(prepare_dataset, remove_columns=['audio'])
    return prep_wvEN

def prepare_wvEN10Fold():
    "Prepares WV English stimuli with vowel labels for 10-fold cross validation."
    
    human_responses = pd.read_csv('../human_vowel_responses.csv')
    human_responses = human_responses[(human_responses['language_indiv'] == 'english') * (human_responses['language_stimuli'] == 'EN')]

    wvEN = Dataset.from_dict({
        'audio': [f'../stimuli_world_vowels/{file}.wav' for file in human_responses['filename']],
        'label': human_responses['#phone'],
    }).cast_column('audio', datasets.Audio()).class_encode_column('label').shuffle()

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    prep_wvEN = wvEN.map(prepare_dataset, remove_columns=['audio'])
    splits = prep_wvEN.train_test_split(test_size=1/10)
    prep_wvEN_folds = [splits['test']]
    
    for i in reversed(range(2, 10)):
        splits = splits['train'].train_test_split(test_size=1/i)
        prep_wvEN_folds.append(splits['test'])
    prep_wvEN_folds.append(splits['train'])
    return DatasetDict({f'fold_{i}': prep_wvEN_folds[i] for i in range(10)})

def prepare_timit_ctc(aligned = False):
    """Prepares TIMIT for CTC sequence classification."""
    timit = datasets.load_dataset('timit_asr', data_dir='../timit/TIMIT')
    timit = timit.remove_columns(['file', 'text', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])
    
    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_remove = ['tcl', 'gcl', 'kcl', 'bcl', 'epi', 'h#', 'dcl', 'q', 'pcl', 'pau']

    def extract_utterances(batch):
        batch['utterance'] = [timit_folding.get(phone, phone) for phone in batch['phonetic_detail']['utterance'] if phone not in timit_remove]
        if aligned:
            batch['start'] = batch['phonetic_detail']['start']
            batch['end'] = batch['phonetic_detail']['stop']
        return batch

    timit = timit.map(extract_utterances, remove_columns=['phonetic_detail'])

    vocab_dict = _get_vocab_dict(timit, 'utterance')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        batch['labels'] = [vocab_dict[phone] for phone in batch['utterance']]
        return batch

    timit = timit.map(prepare_dataset, remove_columns=['audio', 'utterance'])

    return timit, vocab_dict

def prepare_bl_ctc(aligned = False) -> tuple[datasets.DatasetDict, dict[str, int]]:
    """
    Prepares BL-Database and vocab_dict for CTC training.
    
    Returns an aligned dataset if 'aligned' is set to True.
    """
    bl = datasets.load_dataset('../BL-Database/dataset')
    bl = bl['train'].train_test_split() # type: ignore # known to be DatasetDict
    bl = bl.cast_column('audio', datasets.Audio(sampling_rate=16000))

    def extract_utterances(batch):
        batch['phone'] = batch['phonetic_transcription']['phone']
        if aligned:
            batch['start'] = batch['phonetic_transcription']['start']
            batch['end'] = batch['phonetic_transcription']['stop']
        return batch

    bl = bl.map(extract_utterances, remove_columns=['phonetic_transcription'])
    
    vocab_dict = _get_vocab_dict(bl, 'phone')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json', do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        batch['labels'] = processor(text=batch['phone'], is_split_into_words=True, add_special_tokens=False).input_ids
        return batch
    
    bl = bl.map(prepare_dataset, remove_columns=['phone', 'audio'])
    return bl, vocab_dict

def prepare_targets():
    """Extracts TIMIT syllables into a dataset, which is prepared for training."""
    timit = cast(DatasetDict, datasets.load_dataset('timit_asr', data_dir='../timit/TIMIT'))
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in cast(list[str], batch['phonetic_detail']['utterance'])]
        # positions and labels of all C+VC+ sequences (excluding non-target diphthongs)
        target_indices = [i for i, phone in enumerate(utterance) if phone in target_labels.keys()]
        vowel_indices = [i for i, phone in enumerate(utterance) if phone in timit_vowels]
        target_start = [None for i in target_indices]
        target_end = [None for i in target_indices]
        target_utterance = [target_labels[utterance[i]] for i in target_indices]
        for i, target_index in enumerate(target_indices):
            preceding_vowel_index = max(target_indices[:i] + [j for j in vowel_indices if j < target_index] + [0])
            target_start[i] = batch['phonetic_detail']['start'][preceding_vowel_index + 1]
            following_vowel_index = min([j for j in target_indices[i:] if j > target_index] + [j for j in vowel_indices if j > target_index] + [len(utterance)])
            target_end[i] = batch['phonetic_detail']['stop'][following_vowel_index - 1]
        batch['target_start'] = target_start
        batch['target_end'] = target_end
        batch['target_utterance'] = target_utterance    
        return batch
    timit = timit.map(find_targets)

    def generator(batch):
        for row in range(len(batch)):
            sampling_rate = batch[row]['audio']['sampling_rate']
            for target_start, target_end, target_utterance in zip(batch[row]['target_start'], batch[row]['target_end'], batch[row]['target_utterance']):
                yield {
                    'audio_array': batch[row]['audio']['array'][target_start:target_end],
                    'audio_sampling_rate': sampling_rate,
                    'target_utterance': target_utterance
                }
            
    targets_test = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['test']}, split = datasets.Split.TEST)
    targets_train = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['train']}, split = datasets.Split.TRAIN)
    targets = datasets.DatasetDict({'test': targets_test, 'train': targets_train})
    targets.save_to_disk('../timit_targets')

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        batch['input_values'] = feature_extractor(batch['audio_array'], sampling_rate=batch['audio_sampling_rate']).input_values[0]
        batch['labels'] = batch['target_utterance']
        return batch

    prepared_targets = targets.map(prepare_dataset, remove_columns=targets.column_names['train'])
    prepared_targets = prepared_targets.class_encode_column('labels')

    return prepared_targets

def prepare_masked_targets():
    """
    Prepares TIMIT utterances, with additional columns 'sample_start' and 'sample_stop'.
    """
    timit = cast(DatasetDict, datasets.load_dataset('timit_asr', data_dir='../timit/TIMIT'))
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in cast(list[str], batch['phonetic_detail']['utterance'])]
        # positions and labels of all C+VC+ sequences (excluding non-target diphthongs)
        target_indices = [i for i, phone in enumerate(utterance) if phone in target_labels.keys()]
        vowel_indices = [i for i, phone in enumerate(utterance) if phone in timit_vowels]
        target_start = [None for i in target_indices]
        target_end = [None for i in target_indices]
        target_utterance = [target_labels[utterance[i]] for i in target_indices]
        for i, target_index in enumerate(target_indices):
            preceding_vowel_index = max(target_indices[:i] + [j for j in vowel_indices if j < target_index] + [0])
            target_start[i] = batch['phonetic_detail']['start'][preceding_vowel_index + 1]
            following_vowel_index = min([j for j in target_indices[i:] if j > target_index] + [j for j in vowel_indices if j > target_index] + [len(utterance)])
            target_end[i] = batch['phonetic_detail']['stop'][following_vowel_index - 1]
        batch['target_start'] = target_start
        batch['target_end'] = target_end
        batch['target_utterance'] = target_utterance    
        return batch
    timit = timit.map(find_targets)

    def generator(batch):
        for row in range(len(batch)):
            audio_array = batch[row]['audio']['array']
            sampling_rate = batch[row]['audio']['sampling_rate']
            for target_start, target_end, target_utterance in zip(batch[row]['target_start'], batch[row]['target_end'], batch[row]['target_utterance']):
                yield {
                    'audio_array': audio_array,
                    'audio_sampling_rate': sampling_rate,
                    'target_utterance': target_utterance,
                    'target_start': target_start,
                    'target_end': target_end
                }

    
    targets_test = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['test']}, split = datasets.Split.TEST)
    targets_train = Dataset.from_generator(generator, gen_kwargs = {'batch': timit['train']}, split = datasets.Split.TRAIN)
    targets = datasets.DatasetDict({'test': targets_test, 'train': targets_train})
    targets.save_to_disk('../timit_masked_targets')

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    def prepare_dataset(batch):
        batch['input_values'] = feature_extractor(batch['audio_array'], sampling_rate=batch['audio_sampling_rate']).input_values[0]
        batch['labels'] = batch['target_utterance']
        batch['sample_start'] = batch['target_start']
        batch['sample_stop'] = batch['target_end']
        return batch

    masked_targets = targets.map(prepare_dataset, remove_columns=targets.column_names['train'])
    masked_targets = masked_targets.class_encode_column('labels')

    return masked_targets

def prepare_librispeechFR_ctc():
    """
    Prepares Multilingual LibriSpeech French for CTC training. Uses the first 10% of the train set
    """

    train = datasets.load_dataset('facebook/multilingual_librispeech', 'french', split='train[:10%]') # approx. 100 hours
    test = datasets.load_dataset('facebook/multilingual_librispeech', 'french', split='test')
    dev = datasets.load_dataset('facebook/multilingual_librispeech', 'french', split='dev')
    librispeechFR = datasets.DatasetDict({
        'train': train,
        'test': test,
        'dev': dev
    })
    librispeechFR = librispeechFR.remove_columns(
        ['original_path', 'begin_time', 'end_time', 'audio_duration', 'speaker_id', 'chapter_id', 'file', 'id']
    )

    def phonemize(batch): # removes length distinction and lax high front vowel
        text = batch['transcript']
        separator = phonemizer.separator.Separator(word = '', phone = ' ')
        batch['phone'] = [phone.replace('ː', '').replace('ɪ', 'i').split() for phone in phonemizer.phonemize(text, language = 'fr-fr', separator = separator)] # type: ignore # returns a string
        return batch

    librispeechFR = librispeechFR.map(phonemize, batched=True, remove_columns='transcript')
    
    def remove_language_switching(batch): # remove any utterances that contain non-french phonemizations
        to_keep = [i for i, phone in enumerate(batch['phone']) if '(fr)' not in phone]
        return {'to_keep': [to_keep]}
    
    to_keep = librispeechFR.map(remove_language_switching, batched=True, batch_size=-1, remove_columns=librispeechFR['train'].column_names)
    for split in ['train', 'test', 'dev']:
        librispeechFR[split] = librispeechFR[split].select(to_keep[split]['to_keep'][0])
    
    vocab_dict = _get_vocab_dict(librispeechFR, 'phone')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json', do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        with processor.as_target_processor():
            batch['labels'] = processor(batch['phone'], is_split_into_words=True, add_special_tokens=False).input_ids
        return batch

    librispeechFR = librispeechFR.map(prepare_dataset, batched=True, batch_size=250, remove_columns=librispeechFR['train'].column_names)

    return librispeechFR, vocab_dict

def prepare_librispeech_ctc(classic=False, word_boundaries=False):
    """
    Prepares Multilingual LibriSpeech English 10k for CTC training. Uses the first 1% of the train set (approx. 100 hours).
    
    If classic = True, prepares classic LibriSpeech 100h instead.
    """

    if classic:
        librispeech = datasets.load_dataset('openslr/librispeech_asr', verification_mode='no_checks', data_dir='all', data_files={
            'train.clean.100': 'train.clean.100/*.parquet',
            'validation.clean': 'validation.clean/*.parquet'
        })
        librispeech = librispeech.remove_columns(
            ['file', 'id', 'speaker_id', 'chapter_id']
        )
    else:
        librispeech = datasets.load_dataset('parler-tts/mls_eng_10k', verification_mode='no_checks', data_dir='data', data_files={
            'train': [f'train-0000{i}-of-00317.parquet' for i in range(3)],
            'test': 'test-00000-of-00001.parquet',
            'dev': 'dev-00000-of-00001.parquet'
        })
        librispeech = librispeech.remove_columns(
            ['original_path', 'begin_time', 'end_time', 'audio_duration', 'speaker_id', 'book_id']
        )
    
    librispeech = cast(DatasetDict, librispeech)

    folding = {'ə': 'ʌ', 'ɐ': 'ʌ', 'ᵻ': 'ɪ', 'əl': 'l', 'ɚ': 'ɹ', 'n̩': 'n', 'ææ': 'æ', 'ɑ̃': 'ɑ', 'o': 'oʊ', 'x': 'k', 'r': 'ɹ'}

    transcript_column = 'text' if classic else 'transcript' # column name changes
    def phonemize(batch): # removes length marker, ensures rhotic vowels are separate phonemes
        text = batch[transcript_column]
        separator = phonemizer.separator.Separator(word = '|' if word_boundaries else '', phone = ' ')
        batch['phone'] = [[folding.get(phone, phone) for phone in phones.replace('ː', '').replace('ɹ', ' ɹ').replace('ɚ', ' ɚ').replace('ɬ', 'ʃ l').replace('ɔ̃', 'ɔ n').replace('aɪə', 'aɪ ə').replace('ɡʲ', 'ɡ j').replace('ʔ', 'ɾ').replace('ɜ ɹ', 'ɚ').replace('ɜ', 'ɚ').replace('iə', 'j ə').split()] for phones in phonemizer.phonemize(text, separator = separator)] # type: ignore # returns a string
        return batch

    librispeech = librispeech.map(phonemize, batched=True, remove_columns=transcript_column)

    def extract_phones(batch):
        all_phones = sum(batch['phone'], [])
        vocab = list(set(all_phones))
        return {'vocab': [vocab], 'all_phones': [all_phones]}
    
    vocab_dict = _get_vocab_dict(librispeech, 'phone')
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer('vocab.json', do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = [audio['array'] for audio in batch['audio']]
        batch['input_values'] = processor(audio, sampling_rate=batch['audio'][0]['sampling_rate']).input_values
        batch['labels'] = processor(text=batch['phone'], is_split_into_words=True, add_special_tokens=False).input_ids
        return batch

    librispeech = librispeech.map(prepare_dataset, batched=True, batch_size=250, remove_columns=librispeech[list(librispeech.keys())[0]].column_names)

    return librispeech, vocab_dict

def _get_vocab_dict(dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict, text_column: str) -> dict[str, int]:
    """
    Given a dataset with splits and transcriptions, returns a vocab_dict.
    
    Args:
        dataset: A dataset with splits, with at least a column containing
            the text to be converted to a vocabulary.
        text_column: The name of the column containing the text. This can
            be a column of pretokenized lists of strings, or just a
            column of strings. In the latter case, each character
            is treated as a separate token.
    """
    
    vocab_set = set()
    
    def _extract_tokens(batch):
        vocab_set.update(*batch[text_column])
        return None
    
    dataset.map(_extract_tokens, batched = True)
    vocab_list = list(vocab_set)
    vocab_list.extend(['|', '<unk>', '<pad>']) # special tokens
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    return vocab_dict

def audio_to_input_values(dataset: Dataset, feature_extractor) -> Dataset:
    """Removes 'audio' column and adds an 'input_values' column using feature_extractor."""

    def _generate_input_values(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    dataset = dataset.map(_generate_input_values, remove_columns = 'audio')
    return dataset
