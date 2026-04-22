import datasets
from transformers import Wav2Vec2FeatureExtractor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
import numpy
from model_architecture import Wav2Vec2LoanwordsModel, MFCCLoanwordsModel, MFCCLoanwordsConfig, DataCollatorWithPaddingForClassification
from math import ceil
import torch
import os
import traceback

wvENResponses10Fold = datasets.load_from_disk('../prep_wvENResponses10Fold')
wvFRResponses10Fold = datasets.load_from_disk('../prep_wvFRResponses10Fold')

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
accuracy_metric = evaluate.load('../metrics/accuracy')

def compute_metrics(pred):
    pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = numpy.argmax(pred_logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=pred_ids, references=pred.label_ids)
    return {'accuracy': accuracy}

model_configs = {
    'w2v2_max_class_2': {'pretrained_model_name_or_path': '../w2v2', 'use_weighted_layer_sum': True},
    'w2v2fr_max_class_2': {'pretrained_model_name_or_path': '../w2v2fr', 'use_weighted_layer_sum': True},
    'mfcc_max_class_2': {},
}

model_inits = {
    'w2v2_max_class_2': Wav2Vec2LoanwordsModel.from_pretrained,
    'w2v2fr_max_class_2': Wav2Vec2LoanwordsModel.from_pretrained,
    'mfcc_max_class_2': lambda *args, **kwargs: MFCCLoanwordsModel(MFCCLoanwordsConfig(*args, **kwargs)),
}

model_datasets = {'wvENResponses10Fold': wvENResponses10Fold, 'wvFRResponses10Fold': wvFRResponses10Fold}

for model_dataset in model_datasets.keys():
    for model_config in model_configs.keys():
        for k in range(10):
            try:
                model = model_inits[model_config](
                    **model_configs[model_config],
                    id2label = {i: v for i, v in enumerate(model_datasets[model_dataset]['fold_0'].features['label'].names)},
                    label2id = {v: i for i, v in enumerate(model_datasets[model_dataset]['fold_0'].features['label'].names)},
                    temporal_pooling='max',
                    max_pooling_windows=7,
                    classifier_head=True,
                    classifier_hidden=True,
                )

                train_dataset = datasets.concatenate_datasets([model_datasets[model_dataset][f'fold_{i}'] for i in range(10) if i != k])
                test_dataset = model_datasets[model_dataset][f'fold_{k}']
                training_args = TrainingArguments(
                    group_by_length=True,
                    per_device_train_batch_size=32,
                    eval_strategy='steps',
                    num_train_epochs=300,
                    bf16=True,
                    save_steps=1500,
                    eval_steps=1500,
                    logging_steps=1500,
                    learning_rate=1e-4,
                    weight_decay=0.005,
                    warmup_ratio=0.25,
                    save_total_limit=3,
                    push_to_hub=False,
                    output_dir=os.path.expanduser(f'~/scratch/trainer_output_{model_config}_{model_dataset}_cross_{k}'),
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
                trainer.save_model(f'm_{model_config}_{model_dataset}_cross_{k}')
                del trainer
                del model
                del train_dataset
                del test_dataset
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"FAILED: {model_config} on {model_dataset}, fold {k}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue