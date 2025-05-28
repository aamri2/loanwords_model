import pandas, numpy
import torch
import json
from transformers import Wav2Vec2Config, AutoModel, PreTrainedModel, Wav2Vec2FeatureExtractor
from train_sequence_classification import Wav2Vec2WithAttentionClassifier
from datasets import Dataset
from pyctcdecode.decoder import build_ctcdecoder
from typing import Union, Self

model_dir = '../models'
model_name = 'final_model_classification'

model = Wav2Vec2WithAttentionClassifier.from_pretrained(f'{model_dir}/{model_name}')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f'{model_dir}/{model_name}')

# with open(f'{model_dir}/{model_name}/vocab.json') as f:
#     vocab_dict = json.load(f)
# vocab_list = [x[0] for x in sorted(vocab_dict.items(), key = lambda x: x[1])]




def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch['input_values']).unsqueeze(0)
        logits = model(input_values).logits.cpu().detach().numpy()[0]
    
    batch['pred_str'] = [vocab_list[ord(i)] for i in decoder.decode(logits)]
    batch['text'] = [vocab_list[i] for i in batch['labels']]
    return batch

def map_to_result_no_labels(batch):
    audio = batch['audio']
    with torch.no_grad():
        input_values = torch.tensor(processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]).unsqueeze(0)
        logits = model(input_values).logits.cpu().detach().numpy()[0]
    
    batch['pred_str'] = [vocab_list[ord(i)] for i in decoder.decode(logits)]
    batch['logits'] = logits
    return batch

# processor = Wav2Vec2Processor.from_pretrained(f'{model_dir}/{model}')
# model = Wav2Vec2ForCTC.from_pretrained(f'{model_dir}/{model}')
# decoder = build_ctcdecoder([chr(i) for i in range(63)]) # unique two-character sequences



def centre_probabilities(batch):
    """Used for CTC pooling"""
    audio = batch['audio']
    with torch.no_grad():
        input_values = torch.tensor(feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]).unsqueeze(0)
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
    mean_probabilities = numpy.array(probabilities[index]['probabilities']).mean(0)[[range(model.config.num_labels)]]
    normed_probabilities = mean_probabilities * 1/mean_probabilities.sum()
    if response:
        return normed_probabilities[responses.index(response)]
    else:
        return normed_probabilities

def pool(probabilities, responses: Union[list, dict], index, index_names):
    if isinstance(responses, list):
        responses = {response: response for response in responses}
        
    if not isinstance(index[0], tuple):
        index = [(i,) for i in index]
        index_names = [index_names]

    pooled_probabilities = pandas.DataFrame(columns = model.config.id2label.values(), index = ['.'.join(indices) for indices in index])
    for indices in index:
        pooled_probabilities.loc['.'.join(indices)] = probability(probabilities, [value for key, value in responses.items()], **{index_name: i for i, index_name in zip(indices, index_names)})
    
    for column in pooled_probabilities:
        pooled_probabilities[column] = [float(i) for i in pooled_probabilities[column]]
    
    return pooled_probabilities


def probabilities(model, dataset: Dataset, id2label = None):
    """Create DataFrame in long format containing classification probabilities"""

    column_names = [column_name for column_name in dataset.column_names if column_name != 'input_values']

    data = {
        'probabilities': [],
        'classification': [],
        **{column_name: [] for column_name in column_names}
    }
    if not id2label:
        id2label = model.config.id2label if model.config.id2label else {}
    
    for row in dataset.to_iterable_dataset():
        with torch.no_grad():
            input_values = torch.tensor(row['input_values'])#.unsqueeze(0)
            logits = model(input_values)['logits'].cpu().detach()[0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        data['probabilities'].extend(probabilities)
        data['classification'].extend(id2label.values())
        for column_name in column_names:
            data[column_name].extend([row[column_name]] * len(probabilities))
    
    return pandas.DataFrame(data)