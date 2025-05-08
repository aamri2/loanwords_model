from datasets import load_dataset, Dataset
import datasets
import json
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import numpy, pandas
from pyctcdecode import build_ctcdecoder
import random
import matplotlib.pyplot as plt

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
targets.save_to_disk('../datasets/timit_targets')

vocabs = targets.unique('target_utterance')

vocab_list = list(set(vocabs['train']) | set(vocabs['test']))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
with open('vocab.json', 'w') as f:
    json.dump(vocab_dict, f)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

def prepare_dataset(batch):
    batch['input_values'] = feature_extractor(batch['audio_array'], sampling_rate=batch['audio_sampling_rate']).input_values[0]
    batch['labels'] = vocab_dict[batch['target_utterance']]
    return batch

prepared_targets = targets.map(prepare_dataset, remove_columns=targets.column_names['train'])

data_collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)
accuracy_metric = evaluate.load('accuracy')

def compute_metrics(pred):
    pred_logits = pred.predictions
    predictions = numpy.argmax(pred_logits, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=pred.label_ids)
    return {'accuracy': accuracy}

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    '../wav2vec2-base',
    num_labels = len(vocab_dict),
    label2id = vocab_dict,
    id2label = {v: k for k, v in vocab_dict.items()}
)
model.freeze_base_model()

training_args = TrainingArguments(
    group_by_length=True,
    per_device_train_batch_size=32,
    eval_strategy='steps',
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=prepared_targets['train'],
    eval_dataset=prepared_targets['test'],
    processing_class=feature_extractor
)

trainer.train()

# tokenizer.save_pretrained(f'{model_dir}/{model}')
# trainer.save_model(f'{model_dir}/{model}')