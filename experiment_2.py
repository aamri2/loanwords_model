import torch
import datasets
from transformers import Wav2Vec2FeatureExtractor, EarlyStoppingCallback, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Processor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
from model_architecture import Wav2Vec2LoanwordsModel, DataCollatorWithPaddingForClassification, DataCollatorCTCWithPadding
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
# EN CTC
# FR CTC
# EN neural-mean pseudosylls

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

if task < 22 or task == 24: # classifier
    model_config = {'classifier_head': True, 'classifier_hidden': True, 'classifier_hidden_activation_function': 'relu', 'temporal_pooling': 'max', 'max_pooling_windows': 7}
    processor = feature_extractor
    data_collator = DataCollatorWithPaddingForClassification(feature_extractor)
    id2label_getter = lambda dataset: {i: v for i, v in enumerate(dataset[train_split].features['label'].names)}
    label2id_getter = lambda dataset: {v: i for i, v in enumerate(dataset[train_split].features['label'].names)}
    metric = evaluate.load('../metrics/accuracy')
    get_predictions = lambda pred: {'predictions': pred.predictions, 'references': pred.label_ids}

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

    if task == 24: # intervocal consonants
        language = 'EN'
        model_name = 'consonants'
        fold = None
        train_split = 'train'
        eval_split = 'test'
        dataset_name = 'timitEC'
        base_model = 'w2v2-large'
    model_training = f'mean_class_2_{dataset_name}_varHiddenRelu{f"_cross_{fold}" if fold is not None else ""}'

elif task in [22, 23]: # ASR
    model_config = {'ctc_head': True, 'ctc_loss_reduction': 'mean'}
    model_name = 'ASR'
    train_split = 'train'
    eval_split = 'test'
    
    if task == 22:
        language = 'EN'
        dataset_name = 'timit'
        base_model = 'w2v2-large'
    elif task == 23:
        language = 'FR'
        dataset_name = 'bl'
        base_model = 'w2v2fr-large'
    
    tokenizer = Wav2Vec2PhonemeCTCTokenizer(f'../prep_{dataset_name}/vocab.json', do_phonemize=False)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    label2id = tokenizer.get_vocab()
    label2id_getter = lambda x: label2id
    id2label_getter = lambda x: {v: k for k, v in label2id.items()}
    class Metric:
        _metric = evaluate.load('../metrics/cer')
        def compute(self, **kwargs):
            return {'per': self._metric.compute(**kwargs)}
    metric = Metric()
    def get_predictions(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        return {'predictions': processor.batch_decode(pred.predictions), 'references': processor.batch_decode(label_ids, group_tokens = False)}
    model_config |= {'vocab_size': tokenizer.vocab_size}
    model_training = f'ctc_1_{dataset_name}'
    fold = None

model_config |= {'pretrained_model_name_or_path': f'../{base_model}', 'use_weighted_layer_sum': True}
dataset = datasets.load_from_disk(f'../prep_{dataset_name}')
model_config |= {
    'id2label': id2label_getter(dataset),
    'label2id': label2id_getter(dataset),
}

os.environ['TENSORBOARD_LOGGING_DIR'] = os.path.expanduser(f'~/scratch/experiment_2_tensorboard/{language}/{model_name}{f"/cross_{fold}" if fold is not None else ""}')

def compute_metrics(pred):
    return metric.compute(**get_predictions(pred))

model = Wav2Vec2LoanwordsModel.from_pretrained(**model_config)
if model.config.ctc_head:
    model.freeze_feature_encoder()
else:
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
    output_dir=os.path.expanduser(f'~/scratch/trainer_output_{base_model}_{model_training}'),
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
    processing_class=feature_extractor,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(training_args.warmup_steps / training_args.eval_steps), early_stopping_threshold=0.001)], # never stop during warmup
    preprocess_logits_for_metrics=lambda logits, labels: (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1),
)

trainer.train()
trainer.save_model(f'm_{base_model}_{model_training}')
print(f'Saved model m_{base_model}_{model_training}.')
del trainer
del model
del train_dataset
del test_dataset
torch.cuda.empty_cache()