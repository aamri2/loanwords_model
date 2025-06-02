import datasets
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import evaluate
import torch
import numpy
from dataclasses import dataclass
# from typing import List, Dict, Optional, Union
from dataset_handler import prepare_bl_ctc
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding: # from https://huggingface.co/blog/fine-tune-wav2vec2-english
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

if __name__ == '__main__':
    try:
        bl_database = datasets.load_from_disk('../prepared_bl')
        with open('vocab.json') as f:
            vocab_dict = json.load(f)
    except FileNotFoundError:
        bl_database, vocab_dict = prepare_bl_ctc()
        bl_database.save_to_disk('../prepared_bl')
        with open('vocab.json', 'w') as f:
            json.dump(vocab_dict, f)
    
    tokenizer = Wav2Vec2CTCTokenizer('vocab.json')
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
            '../wav2vec2-base-fr',
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
        model.save_pretrained('../wav2vec2-base-fr')
    
    model.freeze_feature_encoder()

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
        warmup_steps=1000,
        save_total_limit=2,
        push_to_hub=False,
        output_dir='../trainer_output'
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=bl_database['train'],
        eval_dataset=bl_database['test'],
        processing_class=processor.feature_extractor, # type: ignore # feature_extractor exists
    )

    trainer.train()