import datasets
from transformers import Wav2Vec2FeatureExtractor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
import numpy
from model_architecture import Wav2Vec2ForCTCWithMaxPooling, DataCollatorWithPaddingForClassification

wvResponses = datasets.load_from_disk('../prep_wvResponses')
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
accuracy_metric = evaluate.load('../metrics/accuracy')

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = numpy.argmax(pred_logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=pred_ids, references=pred.label_ids)
    return {'accuracy': accuracy}

model = Wav2Vec2ForCTCWithMaxPooling.from_pretrained(
    '../models/m_w2v2_ctc_1_timit',
    num_labels=len(wvResponses['train'].features['label'].names)
)

model.freeze_base_model()

training_args = TrainingArguments(
    group_by_length=True,
    per_device_train_batch_size=32,
    eval_strategy='steps',
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_ratio=0.25,
    save_total_limit=2,
    push_to_hub=False,
    output_dir='../trainer_output_w2v2_ctc_1_timit_max_class_3_wvResponses'
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=wvResponses['train'],
    eval_dataset=wvResponses['dev'],
    processing_class=feature_extractor, # type: ignore # feature_extractor exists
    data_collator=DataCollatorWithPaddingForClassification(feature_extractor)
)

trainer.train()
trainer.save_model('m_w2v2_ctc_1_timit_ctc_3_wvResponses')
