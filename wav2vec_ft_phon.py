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

model_dir = '../models'
model = 'final_model_folded'

# timit = load_dataset('timit_asr', data_dir='../timit/TIMIT')
# timit = timit.remove_columns(['file', 'text', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

# def extract_utterances(batch):
#     batch['utterance'] = batch['phonetic_detail']['utterance']
#     return batch

# timit = timit.map(extract_utterances, remove_columns=['phonetic_detail'])

# # identify all phones in data
# def extract_phones(batch):
#     all_phones = [phone for utterance in batch['utterance'] for phone in utterance]
#     vocab = list(set(all_phones))
#     return {'vocab': [vocab], 'all_phones': [all_phones]}

# vocabs = timit.map(extract_phones, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names['train'])
# vocab_list = list(set(vocabs['train']['vocab'][0]) | set(vocabs['test']['vocab'][0]))
# vocab_list.extend(['|', '<unk>', '<pad>'])
# vocab_dict = {v: k for k, v in enumerate(vocab_list)}
# with open('vocab.json', 'w') as f:
#     json.dump(vocab_dict, f)

with open(f'{model_dir}/{model}/vocab.json') as f:
    vocab_dict = json.load(f)
vocab_list = [x[0] for x in sorted(vocab_dict.items(), key = lambda x: x[1])]

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

processor = Wav2Vec2Processor.from_pretrained(f'{model_dir}/{model}')
model = Wav2Vec2ForCTC.from_pretrained(f'{model_dir}/{model}')
decoder = build_ctcdecoder([chr(i) for i in range(63)]) # unique two-character sequences

def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch['input_values']).unsqueeze(0)
        logits = model(input_values).logits.cpu().detach().numpy()[0]
    
    batch['pred_str'] = [vocab_list[ord(i)] for i in decoder.decode(logits)]
    batch['text'] = [vocab_list[i] for i in batch['labels']]
    return batch

# results = timit['test'].map(map_to_result, remove_columns=timit['test'].column_names)

# # evaluate PER
# per_metric.compute(references=[''.join([chr(vocab_dict[i] + 200) for i in row['text']]) for row in results], predictions=[''.join([chr(vocab_dict[i] + 200) for i in row['pred_str']]) for row in results])

# test on other languages
def map_to_result_no_labels(batch):
    audio = batch['audio']
    with torch.no_grad():
        input_values = torch.tensor(processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]).unsqueeze(0)
        logits = model(input_values).logits.cpu().detach().numpy()[0]
    
    batch['pred_str'] = [vocab_list[ord(i)] for i in decoder.decode(logits)]
    batch['logits'] = logits
    return batch

# dutch = load_dataset('facebook/multilingual_librispeech', 'dutch', split='1_hours').remove_columns(['file', 'speaker_id', 'chapter_id', 'id'])
# dutch.map(map_to_result_no_labels)
# import random
# for i in range(5):
#     print(*[' '.join(i[1]) if i[0] == 'pred_str' else i[1] for i in map_to_result_no_labels(dutch[random.randint(0, len(dutch))]).items() if i[0] in ['pred_str', 'transcript']], sep='\n', end='\n\n')


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

def centre_probabilities(batch):
    audio = batch['audio']
    with torch.no_grad():
        input_values = torch.tensor(processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]).unsqueeze(0)
        logits = model(input_values).logits.cpu().detach()[0]
    
    centre = len(logits) // 2
    centre_logits = logits[centre-1:centre+2].mean(0, keepdim = True)
    batch['probabilities'] = torch.nn.functional.softmax(centre_logits, dim=-1)[0][:63] # remove <pad>
    return batch

def select_where(df, **column_values):
    for column, value in column_values.items():
        df = df[df[column] == value]
    return df

def count_where(df, **column_values):
    return len(select_where(df, **column_values))
    
def probability(probabilities, responses, response = None, **column_values):
    index = [i for i in range(len(probabilities)) if all(probabilities[column][i] == value for column, value in column_values.items())]
    mean_probabilities = numpy.array(probabilities[index]['probabilities']).mean(0)[[vocab_list.index(i) for i in responses]]
    normed_probabilities = mean_probabilities * 1/mean_probabilities.sum()
    if response:
        return normed_probabilities[responses.index(response)]
    else:
        return normed_probabilities

world_vowels = world_vowels.map(centre_probabilities)

def pool(probabilities, responses, index, index_names):
    if isinstance(responses, list):
        responses = {response: response for response in responses}
        
    if not isinstance(index[0], tuple):
        index = [(i,) for i in index]
        index_names = [index_names]

    pooled_probabilities = pandas.DataFrame(columns = responses.keys(), index = ['.'.join(indices) for indices in index])
    for indices in index:
        pooled_probabilities.loc['.'.join(indices)] = probability(probabilities, [value for key, value in responses.items()], **{index_name: i for i, index_name in zip(indices, index_names)})
    
    for column in pooled_probabilities:
        pooled_probabilities[column] = [float(i) for i in pooled_probabilities[column]]
    
    return pooled_probabilities

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