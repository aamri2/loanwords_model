import datasets
from transformers import Wav2Vec2FeatureExtractor, EarlyStoppingCallback
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
from model_architecture import Wav2Vec2LoanwordsModel, DataCollatorWithPaddingForClassification
import torch
import os
import sys

task = int(sys.argv[1]) # from slurm array task id
# tasks:
# w2v2-nat:
# [EN neural-mean gold] x 10
# [EN neural-mean pseudosylls]
# [FR neural-mean gold] x 10
# [FR neural-mean psuedosylls]
# CTC
# EN CTC # TODO
# FR CTC # TODO

if task < 22: # classifier
    model_config = {'classifier_head': True, 'classifier_hidden': True, 'classifier_hidden_activation_function': 'relu', 'temporal_pooling': 'mean'}

    if task // 11 == 0: # EN
        language = 'EN'
        pseudosylls_dataset_name = 'timitEV'
        base_model = 'w2v2-large'
    elif task // 11 == 1: # FR
        language = 'FR'
        pseudosylls_dataset_name = 'blEV'
        base_model = 'w2v2fr-large'
    
    if task % 11 < 10: # gold, 10-fold
        model_name = 'gold'
        fold = task % 11
        train_split = f'train_{fold}'
        eval_split = f'dev_{fold}'
        dataset_name = f'wv{language}10Fold'
    elif task % 11 == 10: # pseudosylls, no cross-val
        model_name = 'pseudo-sylls'
        fold = None
        train_split = 'train'
        eval_split = 'test'
        dataset_name = pseudosylls_dataset_name

model_config |= {'pretrained_model_name_or_path': f'../{base_model}', 'use_weighted_layer_sum': True}
dataset = datasets.load_from_disk(f'../prep_{dataset_name}')
model_config |= {
    'id2label': {i: v for i, v in enumerate(dataset[train_split].features['label'].names)},
    'label2id': {v: i for i, v in enumerate(dataset[train_split].features['label'].names)},
}

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
data_collator = DataCollatorWithPaddingForClassification(feature_extractor)

os.environ['TENSORBOARD_LOGGING_DIR'] = os.path.expanduser(f'~/scratch/experiment_2_tensorboard/{language}/{model_name}{f"/cross_{fold}" if fold else ""}')

accuracy_metric = evaluate.load('../metrics/accuracy')

def compute_metrics(pred):
    return accuracy_metric.compute(predictions=pred.predictions, references=pred.label_ids)

model = model_init_fn = Wav2Vec2LoanwordsModel.from_pretrained(**model_config)
model.freeze_base_model()

train_dataset = dataset[train_split]
test_dataset = dataset[eval_split]
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    eval_strategy='steps',
    max_steps=100000,
    bf16=True,
    save_steps=0.05,
    eval_steps=0.05,
    logging_steps=0.05,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=0.25,
    push_to_hub=False,
    output_dir=os.path.expanduser(f'~/scratch/trainer_output_{base_model}_mean_class_{dataset_name}_varHiddenRelu{f"_cross_{fold}" if fold else ""}'),
    report_to='tensorboard',
    train_sampling_strategy='group_by_length',
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=feature_extractor, # type: ignore # feature_extractor exists
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(training_args.warmup_steps / training_args.eval_steps), early_stopping_threshold=0.001)], # never stop during warmup
    preprocess_logits_for_metrics=lambda logits, labels: (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1),
)

trainer.train()
trainer.save_model(f'm_{base_model}_mean_class_{dataset_name}_varHiddenRelu{f"_cross_{fold}" if fold else ""}')
print(f'Saved model m_{base_model}_mean_class_{dataset_name}_varHiddenRelu{f"_cross_{fold}" if fold else ""}.')
del trainer
del model
del train_dataset
del test_dataset
torch.cuda.empty_cache()