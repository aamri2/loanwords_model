import datasets
import json
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import evaluate
import torch
import numpy
from dataset_handler import prepare_librispeechFR_ctc
from model_architecture import DataCollatorCTCWithPadding

try:
    librispeechFR = datasets.load_from_disk('../prep_librispeechFR')
    with open('../prep_librispeechFR/vocab.json') as f:
        vocab_dict = json.load(f)
except FileNotFoundError:
    librispeechFR, vocab_dict = prepare_librispeechFR_ctc()
    librispeechFR.save_to_disk('../prep_librispeechFR')
    with open('../prep_librispeechFR/vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

tokenizer = Wav2Vec2PhonemeCTCTokenizer('../prep_librispeechFR/vocab.json', do_phonemize=False)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
per_metric = evaluate.load('../metrics/cer')

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = numpy.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id # type: ignore # tokenizer exists
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    per = per_metric.compute(predictions=pred_str, references=label_str)
    return {'per': per}

try:
    model = Wav2Vec2ForCTC.from_pretrained(
        '../w2v2fr',
        ctc_loss_reduction='mean',
        pad_token_id=processor.tokenizer.pad_token_id, # type: ignore # tokenizer exists
        vocab_size=len(vocab_dict)
    )
except EnvironmentError:
    model = Wav2Vec2ForCTC.from_pretrained(
        'facebook/wav2vec2-base-fr-voxpopuli-v2',
        ctc_loss_reduction='mean',
        pad_token_id=processor.tokenizer.pad_token_id, # type: ignore # tokenizer exists
        vocab_size=len(vocab_dict)
    )
    model.save_pretrained('../w2v2fr')

model.freeze_feature_encoder()

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
    output_dir='../trainer_output_w2v2fr_ctc_1_librispeechFR'
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=librispeechFR['train'],
    eval_dataset=librispeechFR['dev'],
    processing_class=processor.feature_extractor, # type: ignore # feature_extractor exists
)

trainer.train()