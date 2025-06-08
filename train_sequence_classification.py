import datasets
from transformers import Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
import evaluate
import torch
from torch import nn
import numpy
from dataset_handler import prepare_targets, prepare_masked_targets
from model_architecture import Wav2Vec2ForCTCWithAttentionClassifier, DataCollatorWithFrameMask

if __name__ == '__main__':
    try:
        masked_targets = datasets.load_from_disk('../masked_targets')
    except FileNotFoundError:
        masked_targets = prepare_masked_targets()
        masked_targets.save_to_disk('../masked_targets')
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    accuracy_metric = evaluate.load('accuracy')

    def compute_metrics(pred):
        pred_logits = pred.predictions
        predictions = numpy.argmax(pred_logits, axis=1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=pred.label_ids)
        return {'accuracy': accuracy}

    labels = masked_targets['train'].features['labels'].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    model = Wav2Vec2ForCTCWithAttentionClassifier.from_pretrained(
        '../models/m_w2v2_ctc_1_timit',
        num_labels = len(labels),
        label2id = label2id,
        id2label = id2label
    )
    model.freeze_base_model()

    data_collator = DataCollatorWithFrameMask(processor=feature_extractor, samples_per_frame=model.wav2vec2.config.inputs_to_logits_ratio)


    training_args = TrainingArguments(
        group_by_length=True,
        per_device_train_batch_size=32,
        eval_strategy='steps',
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=5000,
        eval_steps=5000,
        logging_steps=5000,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        push_to_hub=False,
        output_dir='../trainer_output'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=masked_targets['train'],
        eval_dataset=masked_targets['test'],
        processing_class=feature_extractor,
        data_collator=data_collator
    )

    trainer.train()

    trainer.save_model('m_w2v2_ctc_1_timit_attn_class_3_timitMV')

    # tokenizer.save_pretrained(f'{model_dir}/{model}')
    # trainer.save_model(f'{model_dir}/{model}')