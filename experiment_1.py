import datasets
from transformers import Wav2Vec2FeatureExtractor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
import numpy
from model_architecture import Wav2Vec2LoanwordsModel, DataCollatorWithPaddingForClassification
from math import ceil
import torch
import os
import traceback

wvENResponses10Fold = datasets.load_from_disk('../prep_wvENResponses10Fold')

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
accuracy_metric = evaluate.load('../metrics/accuracy')

def compute_metrics(pred):
    pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = numpy.argmax(pred_logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=pred_ids, references=pred.label_ids)
    return {'accuracy': accuracy}

model_configs = {
    'max_class_2_wvENResponses10Fold': {'temporal_pooling': 'max', 'max_pooling_windows': 7, 'classifier_head': True, 'classifier_hidden': True},
    'max_relu_class_2_wvENResponses10Fold_varFiveWindows': {'temporal_pooling': 'max', 'max_pooling_windows': 5, 'preclassifier_activation_function': 'relu', 'classifier_head': True, 'classifier_hidden': True},
    'max_class_2_wvENResponses10Fold_varFiveWindows': {'temporal_pooling': 'max', 'max_pooling_windows': 5, 'classifier_head': True, 'classifier_hidden': True},
    'max_relu_class_2_wvENResponses10Fold_varNoTemporal': {'temporal_pooling': 'max', 'max_pooling_windows': 1, 'preclassifier_activation_function': 'relu', 'classifier_head': True, 'classifier_hidden': True},
    'max_relu_class_2_wvENResponses10Fold_varThreeWindows': {'temporal_pooling': 'max', 'max_pooling_windows': 3, 'preclassifier_activation_function': 'relu', 'classifier_head': True, 'classifier_hidden': True},
    'max_relu_class_2_wvENResponses10Fold': {'temporal_pooling': 'max', 'max_pooling_windows': 7, 'preclassifier_activation_function': 'relu', 'classifier_head': True, 'classifier_hidden': True},
    'max_relu_class_2_wvENResponses10Fold_varFiveWindows_varNoProjection': {'temporal_pooling': 'max', 'max_pooling_windows': 5, 'preclassifier_activation_function': 'relu', 'classifier_head': True, 'classifier_hidden': False},
    'mean_class_2_wvENResponses10Fold': {'temporal_pooling': 'mean', 'classifier_head': True, 'classifier_hidden': True},
}

for model_config in model_configs.keys():
    for k in range(10):
        try:
            model = Wav2Vec2LoanwordsModel.from_pretrained(
                '../w2v2',
                id2label = {i: v for i, v in enumerate(wvENResponses10Fold['fold_0'].features['label'].names)},
                label2id = {v: i for i, v in enumerate(wvENResponses10Fold['fold_0'].features['label'].names)},
                use_weighted_layer_sum=True,
                **model_configs[model_config]
            )

            train_dataset = datasets.concatenate_datasets([wvENResponses10Fold[f'fold_{i}'] for i in range(10) if i != k])
            test_dataset = wvENResponses10Fold[f'fold_{k}']
            training_args = TrainingArguments(
                group_by_length=True,
                per_device_train_batch_size=32,
                eval_strategy='steps',
                num_train_epochs=300,
                bf16=True,
                save_steps=0.1,
                eval_steps=0.1,
                logging_steps=0.1,
                learning_rate=1e-4,
                weight_decay=0.005,
                warmup_ratio=0.25,
                save_total_limit=3,
                push_to_hub=False,
                output_dir=os.path.expanduser(f'~/scratch/trainer_output_{model_config}_cross_{k}'),
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss'
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                processing_class=feature_extractor, # type: ignore # feature_extractor exists
                data_collator=DataCollatorWithPaddingForClassification(feature_extractor)
            )

            trainer.train()
            trainer.save_model(f'm_{model_config}_cross_{k}')
            del trainer
            del model
            del train_dataset
            del test_dataset
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED: {model_config}, fold {k}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue