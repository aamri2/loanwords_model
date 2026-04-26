import datasets
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
import numpy
from model_architecture import Wav2Vec2LoanwordsModel, DataCollatorCTCWithPadding
import torch
import os
import traceback
import json

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
per_metric = evaluate.load('../metrics/cer')

model_configs = {
    'w2v2_ctc_1': {'pretrained_model_name_or_path': '../w2v2'},
    'w2v2_transformer_ctc_2': {'pretrained_model_name_or_path': '../w2v2', 'use_weighted_layer_sum': True, 'transform_hidden_outputs': True},
    'w2v2fr_ctc_1': {'pretrained_model_name_or_path': '../w2v2fr'},
    'w2v2fr_transformer_ctc_2': {'pretrained_model_name_or_path': '../w2v2fr', 'use_weighted_layer_sum': True, 'transform_hidden_outputs': True},
}

model_datasets_by_config = {
    'w2v2_ctc_1': {
        'timit': datasets.load_from_disk('../prep_timit'),
        'timitS': datasets.load_from_disk('../prep_timitS'),
    },
    'w2v2_transformer_ctc_2': {
        'timit': datasets.load_from_disk('../prep_timit'),
        'timitS': datasets.load_from_disk('../prep_timitS'),
    },
    'w2v2fr_ctc_1': {
        'bl': datasets.load_from_disk('../prep_bl'),
        'blS': datasets.load_from_disk('../prep_blS'),
    },
    'w2v2fr_transformer_ctc_2': {
        'bl': datasets.load_from_disk('../prep_bl'),
        'blS': datasets.load_from_disk('../prep_blS'),
    },
}

dataset_vocabs = {}
with open('../prep_timit/vocab.json', encoding='utf-8') as f: dataset_vocabs['timit'] = json.load(f)
with open('../prep_timitS/vocab.json', encoding='utf-8') as f: dataset_vocabs['timitS'] = json.load(f)
with open('../prep_bl/vocab.json', encoding='utf-8') as f: dataset_vocabs['bl'] = json.load(f)
with open('../prep_blS/vocab.json', encoding='utf-8') as f: dataset_vocabs['blS'] = json.load(f)

dataset_processors = {
    'timit': Wav2Vec2Processor(feature_extractor, Wav2Vec2CTCTokenizer('../prep_timit/vocab.json')),
    'timitS': Wav2Vec2Processor(feature_extractor, Wav2Vec2CTCTokenizer('../prep_timitS/vocab.json')),
    'bl': Wav2Vec2Processor(feature_extractor, Wav2Vec2CTCTokenizer('../prep_bl/vocab.json')),
    'blS': Wav2Vec2Processor(feature_extractor, Wav2Vec2CTCTokenizer('../prep_blS/vocab.json')),
}

for model_config in model_configs.keys():
    for model_dataset in model_datasets_by_config[model_config].keys():
        try:
            model = Wav2Vec2LoanwordsModel.from_pretrained(
                **model_configs[model_config],
                id2label = {v: k for k, v in dataset_vocabs[model].items()},
                label2id = dataset_vocabs[model_dataset],
                ctc_head = True,
            )
            if '2' in model_config:
                model.freeze_base_model()
            elif '1' in model_config:
                model.freeze_feature_encoder()
            
            def compute_metrics(pred):
                pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
                pred_ids = numpy.argmax(pred_logits, axis=-1)
                pred.label_ids[pred.label_ids == -100] = dataset_processors[model_dataset].tokenizer.pad_token_id

                pred_str = dataset_processors[model_dataset].batch_decode(pred_ids)
                label_str = dataset_processors[model_dataset].batch_decode(pred.label_ids, group_tokens=False)

                per = per_metric.compute(predictions=pred_str, references=label_str)
                return {'per': per}

            train_dataset = model_datasets_by_config[model_config][model_dataset]['train']
            test_dataset = model_datasets_by_config[model_config][model_dataset]['test']
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
                output_dir=os.path.expanduser(f'~/scratch/trainer_output_{model_config}_{model_dataset}'),
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss'
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                processing_class=dataset_processors[model_dataset],
                data_collator=DataCollatorCTCWithPadding(dataset_processors[model_dataset])
            )

            trainer.train()
            trainer.save_model(f'm_{model_config}_{model_dataset}')
            print(f'Saved model m_{model_config}_{model_dataset}.')
            del trainer
            del model
            del train_dataset
            del test_dataset
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED: {model_config} on {model_dataset}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue