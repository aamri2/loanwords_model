import pandas, numpy
import torch
import json
from transformers import Wav2Vec2Config, AutoModel, PreTrainedModel, Wav2Vec2FeatureExtractor
from model_architecture import Wav2Vec2WithAttentionClassifier
from datasets import Dataset
from pyctcdecode.decoder import build_ctcdecoder
from typing import Union, Self, Optional, Any, Callable
import seaborn
import matplotlib.pyplot as plt
import collections

model_dir = '../models'
model_name = 'm_w2v2_attn_class_2_timitEV'

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

def ctc_wrapper(model: PreTrainedModel, **kwargs) -> Callable[[torch.Tensor], dict[str, torch.Tensor]]:
    """
    Model wrapper for the probabilities pipeline with CTC.
    
    Gets the three middlest frames and averages them.
    """

    def model_wrapper(input_values: torch.Tensor):
        with torch.no_grad():
            logits: torch.Tensor = model(input_values, **kwargs).logits.cpu().detach()[0]
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

def probabilities(model, dataset: Dataset, id2label = None, **kwargs):
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
            logits = model(input_values, **kwargs)['logits'].cpu().detach()[0]

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

def ctc_beam_search_cvc(logits, consonant_ids: list[int], vowel_id2label: dict[int, str], padding_token_id: int, beam_width: int = 1):
    """Custom CTC beam search that looks for a CVC sequence."""

    # Collapse all consonants
    consonant_logits = logits[:, :, consonant_ids].logsumexp(dim = 2)
    vowel_logits = logits[:, :, sorted(list(vowel_id2label.keys()))]
    padding_logits = logits[:, :, padding_token_id]

    # create new logits with only relevant vowels, consonants, and padding token
    logits = torch.cat([
        consonant_logits.unsqueeze(2),
        vowel_logits,
        padding_logits.unsqueeze(2)
    ], dim = 2)
    
    sorted_vowel_labels = [v for k, v in sorted(vowel_id2label.items(), key = lambda x: x[0])]

    consonant = 0 # first ID is consonant token
    vowels = [i + 1 for i in range(len(sorted_vowel_labels))]
    pad = len(sorted_vowel_labels) + 1 # last ID is padding token
    id2label = {
        0: 'C',
        **{(i + 1): v for i, v in enumerate(sorted_vowel_labels)},
        pad: '<pad>'
    }
    
    # beams are formed C V C
    T = logits.shape[1] # time
    S = logits.shape[2] # vocabulary
    NEG_INF = torch.tensor(-float('inf'))
    def make_new_beam():
        fn = lambda: (NEG_INF, NEG_INF)
        return collections.defaultdict(fn)
    beam = [(tuple(), (torch.tensor(0.0), NEG_INF))]
    
    for t in range(T): # loop over time
        # must find in order
        next_beam = make_new_beam()
        for s in range(S): # loop over labels
            p = logits[0, t, s]
            # probabilities for prefix given it
            # does (p_b) or does not (p_nb) end
            # in a pad at time t
            for prefix, (p_b, p_nb) in beam: # loop over beam
                if not prefix and s in vowels:
                    continue # can't start with a vowel
                if len(prefix) >= 2 and s in vowels and s != prefix[-1]:
                    continue # can't have more than one vowel
                if s == pad: # prefix doesn't change, p_b changes
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = torch.stack([n_p_b, p_b + p, p_nb + p]).logsumexp(dim = -1)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue
                end_t  = prefix[-1] if prefix else None # current final token
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t: # update non-padding probability
                    n_p_nb = torch.stack([n_p_nb, p_b + p, p_nb + p]).logsumexp(dim = -1)
                    next_beam[n_prefix] = (n_p_b, n_p_nb)
                
                # no repeated tokens

                if s == end_t: # update unchanged prefix
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = torch.stack([n_p_nb, p_nb + p]).logsumexp(dim = -1)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # sort and trim
        beam = sorted(next_beam.items(), key = lambda x: torch.stack(x[1]).logsumexp(dim = -1).item(), reverse = True)
        beam = beam[:beam_width]

    beam = {prefix[1]: torch.stack(probabilities).logsumexp(dim = -1) for prefix, probabilities in beam if len(prefix) == 3} # remove too-short outputs
    logits = torch.stack([beam[i + 1] for i in range(len(sorted_vowel_labels))]).unsqueeze(0)
    return logits

def ctc_cvcBeamSearch_wrapper(model: PreTrainedModel, consonant_ids: list[int], vowel_id2label: dict[int, str], padding_token_id: int, beam_width: int = 1, **kwargs):
    class ModelWrapper:
        class Config:
            id2label = {}
        config = Config()
        def __init__(self, vowel_id2label):
            sorted_vowel_labels = [v for k, v in sorted(vowel_id2label.items(), key = lambda x: x[0])]
            self.config.id2label = {i: sorted_vowel_labels[i] for i in range(len(sorted_vowel_labels))}

        def __call__(self, input_values):
            with torch.no_grad():
                logits = model(input_values, **kwargs).logits.cpu().detach()
            logits = ctc_beam_search_cvc(logits, consonant_ids=consonant_ids, vowel_id2label=vowel_id2label, padding_token_id=padding_token_id, beam_width=beam_width)
            return {'logits': logits}

    return ModelWrapper(vowel_id2label)


if __name__ == '__main__':
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from tokenizers import decoders
    import torch
    import json
    model = Wav2Vec2ForCTC.from_pretrained('../models/m_w2v2_ctc_1_timit')
    processor = Wav2Vec2Processor.from_pretrained('../models/m_w2v2_ctc_1_timit')

    with open('../models/m_w2v2_ctc_1_timit/vocab.json') as f:
        label2id = json.load(f)

    consonant_ids = [0, 3, 5, 7, 8, 10, 11, 12, 13, 15, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38]
    human_timit_vowels = {'i': 'iy', 'ɪ': 'ih', 'eɪ': 'ey', 'ɛ': 'eh', 'æ': 'ae', 'ɑ': 'aa', 'ʌ': 'ah', 'oʊ': 'ow', 'u': 'uw', 'ʊ': 'uh'}
    vowel_id2label = {label2id[timit_vowel]: human_vowel for human_vowel, timit_vowel in human_timit_vowels.items()}
    padding_token_id = 41

    def pipe(n: int):
        logits = model(torch.tensor([wv[i]['input_values'] for i in [n]])).logits.detach()
        beams = ctc_beam_search_cvc(logits, consonant_ids, vowel_id2label, padding_token_id, beam_width=100)
        return beams, wv[n]['vowel']
        #pred_ids = torch.argmax(logits, dim=-1)
        #output = processor.batch_decode(pred_ids)
        #return logits, pred_ids, output