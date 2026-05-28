import torch
import datasets
from test_dataset_handler import t
from torchaudio.transforms import MFCC
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from model_architecture import DataCollatorWithPaddingForClassification, FormantFeatureExtractor
from probabilities_handler import world_vowel_sort
import pandas as pd
import numpy as np

feature_extractor = Wav2Vec2FeatureExtractor()
data_collator = DataCollatorWithPaddingForClassification(feature_extractor)
formant_extractor = FormantFeatureExtractor()
def mfcc_processor(input_values):
    return MFCC()(input_values).mean(-1) # averages across time

def w2v2_processor(input_values, model):
    with torch.no_grad():
        return torch.stack(model(input_values.unsqueeze(0), output_hidden_states=True)[2], dim=1).squeeze(0).mean((0, 1)) # averages across layers and time

representations = ['mfcc', 'w2v2-nat', 'formant']
languages = ['EN', 'FR']
domains = ['native', 'nonnative']

for language in languages:
    test_ds = t['wv'].dataset.with_format('torch')
    for representation in representations:
        if representation == 'mfcc':
            base_model = 'mfcc'
            feature_processor = mfcc_processor
        elif representation == 'w2v2-nat':
            base_model = 'w2v2-large' if language == 'EN' else f'w2v2{language.lower()}-large'
            model = Wav2Vec2Model.from_pretrained(f'../{base_model}')
            feature_processor = lambda x: w2v2_processor(x, model)
        elif representation == 'formant':
            base_model = 'formant'
            feature_processor = lambda x: formant_extractor(x)['input_values'].squeeze(0).mean(0) # average across time
        test_ds = test_ds.map(lambda batch: {'feature_vector': feature_processor(batch['input_values'])})
        test_feature_vector_norm = test_ds['feature_vector'] / test_ds['feature_vector'].norm(dim=0, keepdim=True)
        for domain in domains:
            for fold in range(10):
                dataset_name = f"wv{language}{'Nonnative' if domain == 'nonnative' else ''}Responses10Fold"
                train_ds = datasets.load_from_disk(f'../prep_{dataset_name}')[f'train_{fold}']
                train_ds = train_ds.with_format('torch').map(lambda batch: {'feature_vector': feature_processor(batch['input_values'])})
                k = int(len(train_ds) ** 0.5) # sqrt(N)-nearest-neighbours
                train_feature_vector_norm = train_ds['feature_vector'] / train_ds['feature_vector'].norm(dim=0, keepdim=True)
                cosine_similarities = torch.tensordot(test_feature_vector_norm, train_feature_vector_norm, dims=([1], [1]))
                k_nearest_neighbours = cosine_similarities.topk(k, dim=1, sorted=False).indices
                labels = train_ds['label'][k_nearest_neighbours]
                label_counts = torch.empty((len(test_ds), len(train_ds.features['label'].names)))
                for i in range(label_counts.shape[1]):
                    label_counts[:, i] = (labels == i).sum(dim=1)
                classification_probabilities = label_counts.softmax(-1)
                p_ds = test_ds
                for i, classification in enumerate(train_ds.features['label'].names):
                    p_ds = p_ds.add_column(classification, classification_probabilities[:, i].numpy())
                p = p_ds.to_pandas()
                p = p.melt(id_vars = ['language', 'vowel', 'file'], value_vars = train_ds.features['label'].names, var_name = 'classification', value_name = 'probabilities')
                p = world_vowel_sort(p)
                p.to_csv(f'probabilities/experiment_1/p_{base_model}_knn_2_{dataset_name}_cross_{fold}_wv.csv')