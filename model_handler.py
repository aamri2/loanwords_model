import pandas, numpy
import torch
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder

model_dir = '../models'
model = 'final_model_folded'

with open(f'{model_dir}/{model}/vocab.json') as f:
    vocab_dict = json.load(f)
vocab_list = [x[0] for x in sorted(vocab_dict.items(), key = lambda x: x[1])]

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

processor = Wav2Vec2Processor.from_pretrained(f'{model_dir}/{model}')
model = Wav2Vec2ForCTC.from_pretrained(f'{model_dir}/{model}')
decoder = build_ctcdecoder([chr(i) for i in range(63)]) # unique two-character sequences

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