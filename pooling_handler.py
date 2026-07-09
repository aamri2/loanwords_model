"""Deals with wrapping models for pooling logits when necessary."""

import torch
from typing import Callable, cast, Any
from spec import PoolingSpec
from model_handler import Model
from dataset_handler import TrainingDataset
import pandas as pd
from ctc_decoder import decode_probabilities, Recursive

class Pooling():
    """Transforms model logits to classification probabilities."""

    spec: PoolingSpec | None

    def __init__(self, spec: str | PoolingSpec | None):
        if spec:
            self.spec = PoolingSpec(spec)
        else:
            self.spec = None # default pooling method

    def as_map(self, model: Model) -> Callable[[dict[str, Any]], dict[str, Any]]:
        pooling_fn = self.get_pooling_function(model)
        return lambda batch: pooling_fn(batch)

    def get_pooling_function(self, model: Model) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if not self.spec:
            return lambda batch: self._softmax_logits(batch['logits'], model) | {k: [i for i in v for j in range(model.model.config.num_labels)] for k, v in batch.items() if k not in ['input_values', 'logits']}
        elif self.spec.method.value == 'decode':
            if self.spec.target.value == 'vowels':
                return lambda batch: self._decode_cvc(batch['logits'], model) | {k: [i for i in v for j in range(len(TrainingDataset(model.spec.output_dataset).vowels))] for k, v in batch.items() if k not in ['input_values', 'logits']}
            if self.spec.target.value == 'consonants':
                return lambda batch: self._decode_vcv(batch['logits'], model) | {k: [i for i in v for j in range(len(TrainingDataset(model.spec.output_dataset).consonants))] for k, v in batch.items() if k not in ['input_values', 'logits']}
        raise NotImplementedError(f"Pooling function for method {self.spec.method} or target {self.spec.target} unknown.")
    
    @staticmethod
    def _softmax_logits(logits: torch.Tensor, model: Model) -> dict[str, Any]:
        """Takes the softmax of the final dimension of the output."""

        return {'probabilities': logits.softmax(-1), 'classification': [model.model.config.id2label[i] for i in range(len(model.model.config.id2label))]}
    
    def _decode_cvc(self, logits: torch.Tensor, model: Model) -> dict[str, Any]:
        vocab = cast(dict[str, Recursive[int]], model.model.config.label2id)
        vocab['C'] = [vocab[C] for C in TrainingDataset(model.spec.output_dataset).consonants]
        vocab['V'] = [vocab[V] for V in TrainingDataset(model.spec.output_dataset).vowels]
        probabilities, classifications = decode_probabilities(symbols=['C', 'V', 'C'], symbol_of_interest=1, logits=logits, vocab=vocab, pad_token_id=model.model.config.pad_token_id, as_strings=True)
        return {'probabilities': probabilities.flatten(), 'classification': classifications * logits.shape[0]}
    
    def _decode_vcv(self, logits: torch.Tensor, model: Model) -> dict[str, Any]:
        vocab = cast(dict[str, Recursive[int]], model.model.config.label2id)
        vocab['C'] = [vocab[C] for C in TrainingDataset(model.spec.output_dataset).consonants]
        vocab['V'] = [vocab[V] for V in TrainingDataset(model.spec.output_dataset).vowels]
        probabilities, classifications = decode_probabilities(symbols=['V', 'C', 'V'], symbol_of_interest=1, logits=logits, vocab=vocab, pad_token_id=model.model.config.pad_token_id, as_strings=True)
        return {'probabilities': probabilities.flatten(), 'classification': classifications * logits.shape[0]}