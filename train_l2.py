import datasets
import json
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TrainingArguments, Trainer
import evaluate
import torch
import numpy
#from dataset_handler import prepare_librispeechFRS_ctc
from model_architecture import DataCollatorCTCWithPadding, Wav2Vec2ForCTCWithTransformerL2

librispeechFRS = datasets.load_from_disk('../prep_librispeechFRS')
with open('../prep_librispeechFRS/vocab.json', encoding='utf-8') as f:
    vocab_dict = json.load(f)

tokenizer = Wav2Vec2PhonemeCTCTokenizer('../prep_librispeechFRS/vocab.json', do_phonemize=False)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
per_metric = evaluate.load('../metrics/cer')

def compute_metrics(pred):
    pred_ids = pred.predictions
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id # type: ignore # tokenizer exists
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    per = per_metric.compute(predictions=pred_str, references=label_str)
    return {'per': per}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)



model = Wav2Vec2ForCTCWithTransformerL2.from_pretrained(
    '../m_w2v2_transformer_ctc_2_librispeechCLS',
    l2_pad_token_id=processor.tokenizer.pad_token_id, # type: ignore # tokenizer exists
    l2_vocab_size=len(vocab_dict)
)

model.freeze_l1_model()

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
    warmup_ratio=0.25,
    save_total_limit=2,
    push_to_hub=False,
    output_dir='~/scratch/trainer_output_w2v2_transformer_ctc_2_librispeechCLS_ctc_5_librispeechFRS'
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    train_dataset=librispeechFRS['train'],
    eval_dataset=librispeechFRS['dev'],
    processing_class=processor.feature_extractor, # type: ignore # feature_extractor exists
)
trainer.train()
trainer.save_model('m_w2v2_transformer_ctc_2_librispeechCLS_ctc_5_librispeechFRS')
processor.save_pretrained('m_w2v2_transformer_ctc_2_librispeechCLS_ctc_5_librispeechFRS')