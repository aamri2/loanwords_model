import datasets
from transformers import Wav2Vec2FeatureExtractor, EarlyStoppingCallback
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
import numpy
from model_architecture import Wav2Vec2LoanwordsModel, DataCollatorWithPaddingForClassification
from math import ceil
import torch
import os
import sys

task = int(sys.argv[1]) # from slurm array task id
# tasks (x 10 each):
# [EN neural-mean native, EN neural-mean nonnative]
# [EN neural-max native, EN neural-max nonnative]
# [FR neural-mean native, FR neural-mean nonnative]
# [FR neural-max native, FR neural-max nonnative]

if (task // 40) % 2 == 0:
    base_model = 'w2v2-large'
    base_dataset = 'wvEN'
    language = 'EN'
elif (task // 40) % 2 == 1:
    base_model = 'w2v2fr-large'
    base_dataset = 'wvFR'
    language = 'FR'

if (task // 20) % 2 == 0:
    model_name = 'mean'
    model_config = {'temporal_pooling': 'mean'}
elif (task // 20) % 2 == 1:
    model_name = 'max'
    model_config = {'temporal_pooling': 'max', 'max_pooling_windows': 7}

if (task // 10) % 2 == 0:
    dataset_var = 'Responses10Fold'
    domain = 'native'
elif (task // 10) % 2 == 1:
    dataset_var = 'NonnativeResponses10Fold'
    domain = 'nonnative'

os.environ['TENSORBOARD_LOGGING_DIR'] = os.path.expanduser(f'~/scratch/experiment_1_tensorboard/{language}/{domain}/{model_name}')

dataset = datasets.load_from_disk(f'../prep_{base_dataset}{dataset_var}')
fold = task % 10

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
accuracy_metric = evaluate.load('../metrics/accuracy')

def compute_metrics(pred):
    pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = pred_logits

    accuracy = accuracy_metric.compute(predictions=pred_ids, references=pred.label_ids)
    return accuracy

model = Wav2Vec2LoanwordsModel.from_pretrained(
    f'../{base_model}',
    id2label = {i: v for i, v in enumerate(dataset['train_0'].features['label'].names)},
    label2id = {v: i for i, v in enumerate(dataset['train_0'].features['label'].names)},
    use_weighted_layer_sum = True,
    classifier_head = True,
    classifier_hidden = True,
    classifier_hidden_activation_function = 'relu',
    **model_config,
)
model.freeze_base_model()

train_dataset = dataset[f'train_{fold}']
test_dataset = dataset[f'dev_{fold}']
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
    output_dir=os.path.expanduser(f'~/scratch/trainer_output_{model_name}_class_{base_dataset}{dataset_var}_varHiddenRelu_cross_{fold}'),
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
    data_collator=DataCollatorWithPaddingForClassification(feature_extractor),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(training_args.warmup_steps / training_args.eval_steps), early_stopping_threshold=0.001)], # never stop during warmup
    preprocess_logits_for_metrics=lambda logits, labels: numpy.argmax(logits[0].cpu(), axis=-1),
)

trainer.train()
trainer.save_model(f'm_{base_model}_{model_name}_class_2_{base_dataset}{dataset_var}_varHiddenRelu_cross_{fold}')
print(f'Saved model m_{base_model}_{model_name}_class_2_{base_dataset}{dataset_var}_varHiddenRelu_cross_{fold}.')
del trainer
del model
del train_dataset
del test_dataset
torch.cuda.empty_cache()