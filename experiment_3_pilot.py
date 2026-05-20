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

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
accuracy_metric = evaluate.load('../metrics/accuracy')

def compute_metrics(pred):
    pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = pred_logits

    accuracy = accuracy_metric.compute(predictions=pred_ids, references=pred.label_ids)
    return {'accuracy': accuracy}

model_configs = {
    'w2v2_max_class_2': {'pretrained_model_name_or_path': '../w2v2'},
    'w2v2fr_max_class_2': {'pretrained_model_name_or_path': '../w2v2fr'},
}

model_inits = {
    'w2v2_max_class_2': Wav2Vec2LoanwordsModel.from_pretrained,
    'w2v2fr_max_class_2': Wav2Vec2LoanwordsModel.from_pretrained,
}

model_datasets_by_config = {
    'w2v2_max_class_2': {
        'wvENResponses10Fold': datasets.load_from_disk('../prep_wvENResponses10Fold'),
        'wvEN10Fold': datasets.load_from_disk('../prep_wvEN10Fold'),
        'timitEV10Fold': datasets.load_from_disk('../prep_timitEV10Fold'),
    },
    'w2v2fr_max_class_2': {
        'wvFRResponses10Fold': datasets.load_from_disk('../prep_wvFRResponses10Fold'),
        'wvFR10Fold': datasets.load_from_disk('../prep_wvFR10Fold'),
        'blEV10Fold': datasets.load_from_disk('../prep_blEV10Fold'),
    },
}

dataset_num_epochs = {
    'wvENResponses10Fold': 300,
    'wvFRResponses10Fold': 300,
    'wvEN10Fold': 1000,
    'wvFR10Fold': 1000,
    'timitEV10Fold': 30,
    'blEV10Fold': 60,
}

for model_config in model_configs.keys():
    for model_dataset in model_datasets_by_config[model_config].keys():
        for k in range(10):
            try:
                model = Wav2Vec2LoanwordsModel.from_pretrained(
                    **model_configs[model_config],
                    id2label = {i: v for i, v in enumerate(model_datasets_by_config[model_config][model_dataset]['fold_0'].features['label'].names)},
                    label2id = {v: i for i, v in enumerate(model_datasets_by_config[model_config][model_dataset]['fold_0'].features['label'].names)},
                    use_weighted_layer_sum=True,
                    temporal_pooling='max',
                    max_pooling_windows=7,
                    classifier_head=True,
                    classifier_hidden=True,
                    mask_time_prob=0,
                )
                if isinstance(model, Wav2Vec2LoanwordsModel):
                    model.freeze_base_model()

                train_dataset = datasets.concatenate_datasets([model_datasets_by_config[model_config][model_dataset][f'fold_{i}'] for i in range(10) if i != k])
                test_dataset = model_datasets_by_config[model_config][model_dataset][f'fold_{k}']
                training_args = TrainingArguments(
                    group_by_length=True,
                    per_device_train_batch_size=32,
                    eval_strategy='steps',
                    num_train_epochs=dataset_num_epochs[model_dataset],
                    bf16=True,
                    save_steps=0.1,
                    eval_steps=0.1,
                    logging_steps=0.1,
                    learning_rate=1e-4,
                    weight_decay=0.005,
                    warmup_ratio=0.25,
                    save_total_limit=3,
                    push_to_hub=False,
                    output_dir=os.path.expanduser(f'~/scratch/trainer_output_{model_config}_{model_dataset}_cross_{k}'),
                    load_best_model_at_end=True,
                    metric_for_best_model='eval_loss',
                    eval_accumulation_steps=20,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    processing_class=feature_extractor, # type: ignore # feature_extractor exists
                    data_collator=DataCollatorWithPaddingForClassification(feature_extractor),
                    preprocess_logits_for_metrics=lambda logits, labels: numpy.argmax(logits[0].cpu(), axis=-1),
                )

                trainer.train()
                trainer.save_model(f'm_{model_config}_{model_dataset}_cross_{k}')
                print(f'Saved model m_{model_config}_{model_dataset}_cross_{k}.')
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