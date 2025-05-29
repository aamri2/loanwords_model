from datasets import load_dataset
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC #, TrainingArguments, Trainer
import evaluate
import torch
from dataclasses import dataclass
# from typing import List, Dict, Optional, Union
import numpy, pandas
import random
import seaborn
import matplotlib.pyplot as plt

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

# tokenizer = Wav2Vec2CTCTokenizer(f'{model_dir}/{model}/vocab.json')
# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# def prepare_dataset(batch):
#     audio = batch['audio']
#     batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
#     batch['labels'] = [vocab_dict[phone] for phone in batch['utterance']]
#     return batch

# timit = timit.map(prepare_dataset, remove_columns=timit.column_names['train'])

# @dataclass
# class DataCollatorCTCWithPadding:
#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True
#     max_length: Optional[int] = None
#     max_length_labels: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     pad_to_multiple_of_labels: Optional[int] = None
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{'input_values': feature['input_values']} for feature in features]
#         label_features = [{'input_ids': feature['labels']} for feature in features]
#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors='pt',
#         )
#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 max_length=self.max_length_labels,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors='pt',
#             )
        
#         labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         batch['labels'] = labels
#         return batch

# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
per_metric = evaluate.load('cer')

# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = numpy.argmax(pred_logits, axis=-1)
#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
#     pred_str = processor.batch_decode(pred_ids)
#     label_str = processor.batch_decode(pred_ids, group_tokens=False)
#     per = per_metric.compute(predictions=pred_str, references=label_str)
#     return {'per': per}

# model = Wav2Vec2ForCTC.from_pretrained(
#     '../wav2vec2-base',
#     ctc_loss_reduction='mean',
#     pad_token_id=processor.tokenizer.pad_token_id,
#     vocab_size=len(vocab_list)
# )
# model.freeze_feature_encoder()

# training_args = TrainingArguments(
#     group_by_length=True,
#     per_device_train_batch_size=32,
#     eval_strategy='steps',
#     num_train_epochs=30,
#     fp16=True,
#     gradient_checkpointing=True,
#     save_steps=500,
#     eval_steps=500,
#     logging_steps=500,
#     learning_rate=1e-4,
#     weight_decay=0.005,
#     warmup_steps=1000,
#     save_total_limit=2,
#     push_to_hub=False,
# )

# trainer = Trainer(
#     model=model,
#     data_collator=data_collator,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=timit['train'],
#     eval_dataset=timit['test'],
#     tokenizer=processor.feature_extractor,
# )

# trainer.train()

# tokenizer.save_pretrained(f'{model_dir}/{model}')
# trainer.save_model(f'{model_dir}/{model}')