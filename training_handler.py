"""Automates the training process based on a ModelSpec."""

from typing import overload
from dataset_handler import TrainingDataset
from spec import ModelSpec, TrainingSpec, TrainingDatasetSpec, BaseSpec
from model_handler import Model
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from base_model_handler import Base

def train(model_spec: str | BaseSpec | ModelSpec, training_spec: str | TrainingSpec):
    """
    Given a ModelSpec or BaseSpec and a TrainingSpec,
    sets up everything necessary to train the base model
    according to the TrainingSpec.
    """
    
    training_spec = TrainingSpec(training_spec)
    if isinstance(model_spec, str):
        try:
            base_model = Base(model_spec)
        except (ValueError, TypeError):
            base_model = Model(model_spec)

    training_dataset = TrainingDataset(training_spec.training_dataset)
    model = get_pretrained_model(base_model = base_model, training_spec = training_spec)
    tokenizer = get_tokenizer(base_model = base_model, training = training_spec)
    feature_extractor = get_feature_extractor(base_model = base_model)


def get_pretrained_model(base_model: Base | Model, training_spec: TrainingSpec) -> PreTrainedModel:
    """Returns the necessary pretrained model for training."""
    
    if isinstance(base_model, Base):
        model_class = Model.get_model_class(ModelSpec(base = base_model, training = training_spec))
        pretrained_model = model_class.from_pretrained(base_model.path)
    elif isinstance(base_model, Model):
        pretrained_model = base_model.add_layer(training_spec.layer)
    else:
        raise TypeError(f"model must be a Base or Model, not a {model_spec.__class__.__name__}.")
    
    return pretrained_model

def get_tokenizer(base_model: Base | Model, training: TrainingSpec) -> PreTrainedTokenizer:
    """Gets the appropriate tokenizer for training the model."""

    if isinstance(base_model, Base):
        tokenizer_class = base_model.tokenizer_class
    elif isinstance(base_model, Model):
        tokenizer_class = Base(base_model.spec.base).tokenizer_class
    
    tokenizer = tokenizer_class(TrainingDataset(training.training_dataset).vocab_path)
    return tokenizer

def get_feature_extractor(base_model: Base | Model):
    """Gets the appropriate feature extractor for training the model."""

    if isinstance(base_model, Base):
        feature_extractor = base_model.feature_extractor
    elif isinstance(base_model, Model):
        feature_extractor = Base(base_model.spec.base).tokenizer_class
    else:
        raise TypeError(f"base_model must be a Base or a Model, not a {base_model.__class__.__name__}.")
    
    return feature_extractor