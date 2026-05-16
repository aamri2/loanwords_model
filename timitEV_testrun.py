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
import traceback

timitEV = datasets.load_from_disk('../prep_timitEV')

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
accuracy_metric = evaluate.load('../metrics/accuracy')

def compute_metrics(pred):
    pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = pred_logits

    accuracy = accuracy_metric.compute(predictions=pred_ids, references=pred.label_ids)
    return accuracy

model = Wav2Vec2LoanwordsModel.from_pretrained(
    '../w2v2-large',
    id2label = {i: v for i, v in enumerate(timitEV['train'].features['label'].names)},
    label2id = {v: i for i, v in enumerate(timitEV['train'].features['label'].names)},
    use_weighted_layer_sum = True,
    temporal_pooling = 'mean',
    classifier_head = True,
    classifier_hidden = True,
    classifier_hidden_activation_function = 'relu',
)
model.freeze_base_model()

train_dataset = timitEV['train']
test_dataset = timitEV['test']
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    eval_strategy='steps',
    num_train_epochs=100,
    bf16=True,
    save_steps=0.05,
    eval_steps=0.05,
    logging_steps=0.05,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=0.25,
    save_total_limit=3,
    push_to_hub=False,
    output_dir=os.path.expanduser(f'~/scratch/trainer_output_max_class_relu_timitEV'),
    report_to='tensorboard',
    train_sampling_strategy='group_by_length',
    # metric_for_best_model='eval_loss',
    # load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=feature_extractor, # type: ignore # feature_extractor exists
    data_collator=DataCollatorWithPaddingForClassification(feature_extractor),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    preprocess_logits_for_metrics=lambda logits, labels: numpy.argmax(logits[0].cpu(), axis=-1),
)

trainer.train()
trainer.save_model(f'm_w2v2_max_class_relu_timitEV')
print(f'Saved model m_w2v2_max_class_relu_timitEV.')
# del trainer
# del model
# del train_dataset
# del test_dataset
# torch.cuda.empty_cache()