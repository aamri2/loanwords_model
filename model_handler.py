import pandas, numpy
import torch
import json
from transformers import Wav2Vec2Config, AutoModel, PreTrainedModel, Wav2Vec2FeatureExtractor
from train_sequence_classification import Wav2Vec2WithAttentionClassifier
from datasets import Dataset
from pyctcdecode.decoder import build_ctcdecoder
from typing import Union, Self, Optional, Any, Callable
import seaborn
import matplotlib.pyplot as plt

model_dir = '../models'
model_name = 'final_model_classification'

model = Wav2Vec2WithAttentionClassifier.from_pretrained(f'{model_dir}/{model_name}')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f'{model_dir}/{model_name}')




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

def ctc_wrapper(model: PreTrainedModel) -> Callable[[torch.Tensor], dict[str, torch.Tensor]]:
    """
    Model wrapper for the probabilities pipeline with CTC.
    
    Gets the three middlest frames and averages them.
    """

    def model_wrapper(input_values: torch.Tensor):
        with torch.no_grad():
            logits: torch.Tensor = model(input_values).logits.cpu().detach()[0]
        centre = len(logits) // 2
        centre_logits = logits[centre-1:centre+2].mean(0, keepdim = True)
        return {'logits': centre_logits}
    
    return model_wrapper

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

def pool(probabilities: pandas.DataFrame, *args: str, **kwargs: Any):
    """
    Returns a pivot table in wide format.
    
    Args:
        *args: Columns to pool across.
        **kwargs: Columns to filter by.
    """

    if kwargs:
        filter = zip(*[probabilities[column] == value for column, value in kwargs.items()])
        probabilities = probabilities[[all(row) for row in filter]]
    
    pooled_probabilities = probabilities.pivot_table(values = 'probabilities', columns = 'classification', index = args, sort = False)
    return pooled_probabilities

def audio_to_input_values(dataset: Dataset, feature_extractor):
    """Removes 'audio' column and adds an 'input_values' column using feature_extractor."""

    def _generate_input_values(batch):
        audio = batch['audio']
        batch['input_values'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])['input_values'][0]
        return batch

    dataset = dataset.map(_generate_input_values, remove_columns = 'audio')
    return dataset

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
            input_values = torch.tensor(row['input_values']).unsqueeze(0)
            logits = model(input_values)['logits'].cpu().detach()[0]

        if id2label:
            logits_to_keep = []
            for id in id2label.keys(): # allows for selecting certain classifications only
                logits_to_keep.append(logits[id])
            probabilities = torch.nn.functional.softmax(torch.Tensor(logits_to_keep), dim=-1)
            data['classification'].extend(id2label.values())
        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        data['probabilities'].extend(probabilities)
        for column_name in column_names:
            data[column_name].extend([row[column_name]] * len(probabilities))
    
    return pandas.DataFrame(data).astype({'probabilities': float})

def heatmap(probabilities: Union[pandas.DataFrame, list[pandas.DataFrame]], *args: str, **kwargs: Any):
    """Uses pool to create a seaborn heatmap."""
    if isinstance(probabilities, list):
        seaborn.heatmap(pool(probabilities[0], *args, **kwargs), cmap = 'crest', square = True)
        for probability in probabilities[1:]:
            plt.figure()
            seaborn.heatmap(pool(probability, *args, **kwargs), cmap = 'crest', square = True)
    else:
        seaborn.heatmap(pool(probabilities, *args, **kwargs), cmap = 'crest', square = True)
    plt.show()