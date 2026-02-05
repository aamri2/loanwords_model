"""
Contains the Model class, which includes tools for everything
between loading a model and getting its (properly-shaped) logits
with an accompanying vocabulary. Uses the Spec framework to
manage the relevant specifications.
"""

from spec import ModelSpec, BaseSpec, LayerSpec, TrainingSpec, _SEPARATOR
from model_architecture import *
import pandas, numpy
import scipy
import sklearn
import seaborn.objects
import torch
import json
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config, AutoModel, Wav2Vec2FeatureExtractor
from transformers.modeling_utils import PreTrainedModel
from datasets import Dataset
from typing import Union, Self, Optional, Any, Callable
import seaborn
import matplotlib.pyplot as plt
import collections
from base_model_handler import Base

_MODEL_PATH = '../models/'
_MODEL_PREFIX = 'm'

class Model():
    """
    Uses ModelSpec to load and parse models.
    Will ultimately partially automate the process
    of training new models if it cannot find them.
    """

    spec: ModelSpec
    path: str
    model: PreTrainedModel
    vocab: dict[str, int]

    def __init__(self, spec: str | ModelSpec, path: str | None = None):
        self.spec = ModelSpec(spec)
        self.path = path if path else f'{_MODEL_PATH}{_MODEL_PREFIX}{_SEPARATOR}{self.spec}'
        self.model = self.load_model()
        self.vocab = self.get_model_vocab()
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
    
    @classmethod
    def get_model_class(cls, spec: ModelSpec) -> type[PreTrainedModel]:
        """
        Given the layers of the model specification, returns the appropriate
        Huggingface model class.
        """

        layers = spec.layers
        if layers[0].architecture == 'Wav2Vec2':
            if len(layers) >= 2 and layers[1].value == 'ctc':
                if len(layers) == 2:
                    return Wav2Vec2ForCTC
                elif len(layers) == 4:
                    if layers[3].value == 'class':
                        if layers[2].value == 'attn':
                            return Wav2Vec2ForCTCWithAttentionClassifier
                        elif layers[2].value == 'max':
                            return Wav2Vec2ForCTCWithMaxPooling
                elif len(layers) == 5 and layers[3].value == 'relu' and layers[4].value == 'class':
                    if layers[2].value == 'tempmax':
                        return Wav2Vec2ForCTCWithTemporalMaxPoolingReLU
                    elif layers[2].value == 'max':
                        if isinstance(spec.training, tuple):
                            training = spec.training[-1]
                        else:
                            training = spec.training
                        if isinstance(training.training_var, tuple):
                            training_var = training.training_var
                        elif training.training_var:
                            training_var = (training.training_var,)
                        else:
                            training_var = tuple()
                        if 'varHidden' in [var.value for var in training_var]:
                            return Wav2Vec2WithMaxPoolingReLU
                        else:
                            return Wav2Vec2ForCTCWithMaxPoolingReLU
                    elif layers[2].value == 'hiddenmax':
                        return Wav2Vec2ForCTCWithHiddenMaxPoolingReLU
            elif len(layers) == 3 and layers[1].value == 'attn' and layers[2].value == 'class':
                return Wav2Vec2WithAttentionClassifier
            elif len(layers) == 4 and layers[1].value == 'max' and layers[2].value == 'relu' and layers[3].value == 'class':
                return Wav2Vec2WithMaxPoolingReLU
            elif len(layers) >= 3 and layers[1].value == 'transformer' and layers[2].value == 'ctc':
                if len(layers) == 3:
                    return Wav2Vec2ForCTCWithTransformer
                elif len(layers) == 4 and layers[3].value == 'ctc':
                    raise NotImplementedError("Loading function can't handle passing kwargs yet, which are required here.")
                    return Wav2Vec2ForCTCWithTransformerL2
        raise NotImplementedError(f"Model architecture for {layers} unknown.")

    def load_model(self) -> PreTrainedModel:
        """Loads a model given a specification."""

        model_class = self.get_model_class(self.spec)
        return model_class.from_pretrained(self.path)

    def get_model_vocab(self) -> dict[str, int]:
        """Loads a model's vocabulary given a specification."""

        with open(self.path + '/vocab.json', encoding = 'utf-8') as f:
            vocab = json.load(f)
        return vocab
    
    def add_layer(self, layer_spec: str | LayerSpec | tuple[LayerSpec, ...]) -> PreTrainedModel:
        """
        Given a new layer, loads and returns a model with the layer added.
        """

        if isinstance(layer_spec, tuple):
            layer_spec = _SEPARATOR.join(str(layer_spec_i) for layer_spec_i in layer_spec)
        
        new_spec = f'{self.spec}{_SEPARATOR}{layer_spec}'
        model_class = self.get_model_class(ModelSpec(new_spec))
        return model_class.from_pretrained(self.path)

    def as_map(self) -> Callable:
        """Returns a map function that adds logits to a dataset with an 'audio' column."""

        feature_extractor = Base(self.spec.base).feature_extractor

        def apply_model(batch):
            with torch.no_grad():
                if 'audio' in batch:
                    audio = batch['audio']
                else:
                    audio = batch['input_values']
                input = feature_extractor(audio, sampling_rate=16000, return_tensors='pt', padding=True)
                batch['logits'] = self.model(**input).logits
                return batch
        
        return apply_model

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

def mae(probabilities_p: pandas.DataFrame, probabilities_q: pandas.DataFrame, *args: str, **kwargs: Any) -> float:
    """Returns the mean absolute log error between the pooled probabilities."""

    p = pool(probabilities_p, *args, **kwargs).to_numpy()
    q = pool(probabilities_q, *args, **kwargs).to_numpy()
    MAE = numpy.mean(numpy.abs(p - q)).item()
    return MAE

def pairwise_cosine_similarity(probabilities: pandas.DataFrame, by: str, *args: str, **kwargs: Any):
    """Returns pairwise cosine similarities, with a matrix of 'by'."""

    p = pool(probabilities, by, *args, **kwargs)
    cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(p)
    return pandas.DataFrame(cosine_similarity, index=p.index, columns=p.index)

def pairwise_cosine_similarity_heatmap(probabilities: Union[pandas.DataFrame, list[pandas.DataFrame]], by: str, *args: str, **kwargs: Any):
    """Heatmap of pairwise cosine similarities, with a matrix of 'by'."""

    if not isinstance(probabilities, list):
        probabilities = [probabilities]
    for probability in probabilities:
        plt.figure()
        p = pairwise_cosine_similarity(probability, by, *args, **kwargs)
        seaborn.heatmap(p, cmap = 'crest', square = True, vmin = 0, vmax = 1)
    plt.show()

def jensen_shannon_divergence(probabilities_p: pandas.DataFrame, probabilities_q: pandas.DataFrame, *args: str, **kwargs: Any) -> float:
    """Returns the row-wise Jensen-Shannon divergences between the pooled probabilities."""

    p = pool(probabilities_p, *args, **kwargs).to_numpy()
    q = pool(probabilities_q, *args, **kwargs).to_numpy()
    m = (p + q)/2 # mixture distribution

    d_p_m = (p*(numpy.log(p.clip(min=1e-16)) - numpy.log(m.clip(min=1e-16)))).sum(1)
    d_q_m = (q*(numpy.log(q.clip(min=1e-16)) - numpy.log(m.clip(min=1e-16)))).sum(1)
    js_div = 0.5*(d_p_m + d_q_m)
    return js_div

def mean_jensen_shannon_divergence(probabilities_p: pandas.DataFrame, probabilities_q: pandas.DataFrame, *args: str, **kwargs: Any) -> float:
    """Returns the mean row-wise Jensen-Shannon divergences between the pooled probabilities."""

    js_div = jensen_shannon_divergence(probabilities_p, probabilities_q, *args, **kwargs)
    return sum(js_div)/len(js_div)


def mutual_information(probabilities_p: pandas.DataFrame, probabilities_q: pandas.DataFrame, *args: str, **kwargs: Any) -> float:
    """Returns the mutual information between the pooled probabilities."""

    # H(x, y)
    #joint_entropy = scipy.spatial.distance.jensenshannon(probabilities_p, probabilities_q)
    mutual_information = torch.nn.functional.kl_div(torch.log(torch.tensor(probabilities_p.values)), torch.tensor(probabilities_q.values)).item()
    # mutual_information = 

def diffmap(probabilities_p: pandas.DataFrame, probabilities_q: pandas.DataFrame, *args: str, **kwargs: Any):
    """Displays a heatmap of the squared difference between the two probabilities."""

    p = pool(probabilities_p, *args, **kwargs)
    q = pool(probabilities_q, *args, **kwargs)
    diff = p - q
    diff = numpy.abs(diff)
    seaborn.heatmap(diff, cmap = 'crest', square = True, vmax=1, vmin=0)
    plt.text(0.8, 0.8, f'MAE: {mae(probabilities_p, probabilities_q)}')
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

def ctc_cvc_wrapper(model: PreTrainedModel, consonant_ids: list[int], vowel_id2label: dict[int, str], padding_token_id: int, beam_width: int = 1, **kwargs):
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