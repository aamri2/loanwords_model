from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk
import datasets
import json
import pandas
from typing import Optional, Union
from collections.abc import Callable
import phonemizer.separator
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PhonemeCTCTokenizer
import phonemizer
from pyacoustics.speech_filters import speech_shaped_noise
import numpy as np
from torchaudio.functional import forced_align, merge_tokens


class PhoneDataset(Dataset):
    """Extends dataset with useful attributes and functions for phonetic or phonemic transcriptions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO


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
        log_probs = model(input_values).logits
        targets = torch.tensor([batch['labels']])
        tokens, scores = forced_align(log_probs, targets, blank=model.config.pad_token_id)
        merged_tokens = merge_tokens(tokens[0], scores[0])
        start, end = zip(*[(token_span.start, token_span.end) for token_span in merge_tokens(tokens[0], scores[0])])
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

def noisy_and_substrings(aligned_dataset, snr, samples_to_frames_ratio, substring_length = None, substring_ratio = 0.3, noisy_ratio = 0.5, substring_subset_ratio = 0.5):
    """Helper function makes noisy and substring split of aligned dataset."""

    if isinstance(aligned_dataset, DatasetDict):
        new_dataset = DatasetDict()
        for key, dataset in aligned_dataset.items():
            new_dataset[key] = noisy_and_substrings(dataset, snr, samples_to_frames_ratio, substring_length, substring_ratio, noisy_ratio, substring_subset_ratio)
        return new_dataset
    
    random_indices = np.random.choice(range(len(aligned_dataset)), size=(len(aligned_dataset,)), replace=False)

    # make noisy subset
    if noisy_ratio > 0:
        noisy_dataset = make_noisy(
            aligned_dataset.select(random_indices[:int(noisy_ratio*len(aligned_dataset))]), snr
        )
    else:
        noisy_dataset = Dataset.from_dict({'input_values': [], 'labels': [], 'start': [], 'end': []})
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

def prepare_wvEN():
    """Prepares World Vowels English stimuli."""
    
    wv = pandas.read_csv('WorldVowels_stimuli.csv')
    files = ['../wv/' + i for i in wv[wv['language'] == 'EN']['#file_extract']]
    phone = list(wv[wv['language'] == 'EN']['#phone'])
    wvEN = datasets.Dataset.from_dict({'audio': files, 'labels': phone}, split=datasets.Split.TRAIN)\
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



def prepare_timit_ctc(): #TODO: return
    """Prepares TIMIT for CTC sequence classification."""
    timit = load_dataset('timit_asr', data_dir='../timit/TIMIT')
    timit = timit.remove_columns(['file', 'text', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    def extract_utterances(batch):
        batch['utterance'] = batch['phonetic_detail']['utterance']
        return batch

    timit = timit.map(extract_utterances, remove_columns=['phonetic_detail'])

    # identify all phones in data
    def extract_phones(batch):
        all_phones = [phone for utterance in batch['utterance'] for phone in utterance]
        vocab = list(set(all_phones))
        return {'vocab': [vocab], 'all_phones': [all_phones]}

    vocabs = timit.map(extract_phones, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names['train'])
    vocab_list = list(set(vocabs['train']['vocab'][0]) | set(vocabs['test']['vocab'][0]))
    vocab_list.extend(['|', '<unk>', '<pad>'])
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer(f'{model_dir}/{model}/vocab.json')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        batch['labels'] = [vocab_dict[phone] for phone in batch['utterance']]
        return batch

    timit = timit.map(prepare_dataset, remove_columns=timit.column_names['train'])

    return timit

def prepare_bl_ctc(aligned = False) -> tuple[datasets.DatasetDict, dict[str, int]]:
    """
    Prepares BL-Database and vocab_dict for CTC training.
    
    Returns an aligned dataset if 'aligned' is set to True.
    """
    bl = load_dataset('../BL-Database/dataset')
    bl = bl['train'].train_test_split() # type: ignore # known to be DatasetDict
    bl = bl.cast_column('audio', datasets.Audio(sampling_rate=16000))

    def extract_utterances(batch):
        batch['phone'] = batch['phonetic_transcription']['phone']
        if aligned:
            batch['start'] = batch['phonetic_transcription']['start']
            batch['end'] = batch['phonetic_transcription']['stop']
        return batch

    bl = bl.map(extract_utterances, remove_columns=['phonetic_transcription'])
    
    # identify phones in data
    def extract_phones(batch):
        all_phones = sum(batch['phone'], [])
        vocab = list(set(all_phones))
        return {'vocab': [vocab], 'all_phones': [all_phones]}
    
    vocabs = bl.map(extract_phones, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=bl['train'].column_names)
    vocab_list = list(set(vocabs['train']['vocab'][0]) | set(vocabs['test']['vocab'][0]))
    vocab_list.extend(['|', '<unk>', '<pad>'])
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
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
    timit = load_dataset('timit_asr', data_dir='../timit/TIMIT')
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in batch['phonetic_detail']['utterance']]
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
    timit = load_dataset('timit_asr', data_dir='../timit/TIMIT')
    timit = timit.remove_columns(['file', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    timit_folding = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'ux': 'uw'} # did not merge zh/sh
    timit_vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er', 'ix']
    target_labels = {'iy': 'i', 'ih': 'ɪ', 'ey': 'eɪ', 'eh': 'ɛ', 'ae': 'æ', 'aa': 'ɑ', 'ah': 'ʌ', 'ow': 'oʊ', 'uw': 'u', 'uh': 'ʊ'} # desired label: timit label

    def find_targets(batch):
        # fold phones
        utterance = [timit_folding.get(phone, phone) for phone in batch['phonetic_detail']['utterance']]
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

    def extract_phones(batch):
        all_phones = sum(batch['phone'], [])
        vocab = list(set(all_phones))
        return {'vocab': [vocab], 'all_phones': [all_phones]}
    
    vocabs = librispeechFR.map(extract_phones, batched=True, batch_size=-1, remove_columns=librispeechFR['train'].column_names) # type: ignore
    vocab_list = list(set(vocabs['train']['vocab'][0]) | set(vocabs['test']['vocab'][0]) | set(vocabs['dev']['vocab'][0])) # type: ignore
    vocab_list.extend(['|', '<unk>', '<pad>'])
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
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


# class DatasetLoader:
#     """A wrapper for info needed to dynamically load and prepare a dataset or multiple datasets.

#     Attributes:
#         name: A string or list of strings
#     """

#     def __init__(self, name: Union[str, List[str]], data_dir: Optional[str] = None, from_disk = False):
#         self.name = name
#         self.from_disk = True


def prepare(dataset: Union[Dataset, DatasetDict], input: Union[str, list[str]], output: Union[str, list[str]]) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """Prepares a Dataset for training.

    Given a Dataset, it prepares the desired inputs and outputs
    in a valid format for training.

    Args:
        dataset: The dataset to prepare.
        input: The column containing the input values. A list of
            strings is treated as a nested column.
        output: The column containing the output values. A list
            of strings is treated as a nested column.
        TODO:
        input_map: A function applied to input values...
    
    Returns:
        A prepared dataset with two columns: 'input_values' and
        'labels'.
    """

    if isinstance(input, str):
        input = [input]
    
    if isinstance(output, str):
        output = [output]

    columns_to_remove: list[str] = [column_name for column_name in dataset.column_names if column_name not in input[0] and column_name not in output[0]]
    dataset = dataset.remove_columns(columns_to_remove)

    return dataset

