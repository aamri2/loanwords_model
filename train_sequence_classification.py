from datasets import load_dataset, Dataset
import datasets
from transformers import Wav2Vec2FeatureExtractor, TrainingArguments, Trainer, Wav2Vec2Model, Wav2Vec2PreTrainedModel, BatchFeature
import evaluate
import torch
from torch import nn
import numpy
from dataset_handler import prepare_targets, prepare_masked_targets
from dataclasses import dataclass
from typing import Any
#offline = False # set to True for DRAC jobs; requires prepared dataset

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states, attention_mask=None):
        scores = self.attention(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        weighted_sum = torch.sum(hidden_states * weights, dim=1)
        return weighted_sum


class Wav2Vec2WithAttentionClassifier(Wav2Vec2PreTrainedModel):
    """
    Custom Wav2Vec2 sequence classification with a simple attention head.
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.attention_pooling = AttentionPooling(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    
    def freeze_base_model(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        

    def forward(self, input_values, attention_mask=None, frame_mask=None, labels = None):
        """
        Args:
            frame_mask: This mask is only applied after the wav2vec2,
                but before the attention pooling. This can be used,
                for example, to classify part of a larger sequence.
        """
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if frame_mask is not None:
            # adjust to actual length from estimate
            if frame_mask.shape[1] < hidden_states.shape[1]:
                frame_mask = torch.nn.functional.pad(frame_mask, (0, hidden_states.shape[1] - frame_mask.shape[1]), value = 0.0)
            elif frame_mask.shape[1] > hidden_states.shape[1]:
                frame_mask = frame_mask[:, :hidden_states.shape[1]]
            
            frame_mask = frame_mask.unsqueeze(-1).type_as(hidden_states)
            hidden_states = hidden_states * frame_mask

        pooled_output = self.attention_pooling(hidden_states)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                
                elif self.config.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                
                else:
                    self.config.problem_type = "multi_label_classification"
            

            if self.config.problem_type == "regression":
                loss_fn = nn.MSELoss()
            
            elif self.config.problem_type == "single_label_classification":
                loss_fn = nn.CrossEntropyLoss()
            
            elif self.config.problem_type == "multi_label_classification":
                loss_fn = nn.BCEWithLogitsLoss()
            
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}


@dataclass
class DataCollatorWithFrameMask:
    """Pads and returns frame mask."""

    processor: Wav2Vec2FeatureExtractor
    samples_per_frame: int
    sampling_rate: int = 16000

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        labels = [f['labels'] for f in features] if 'labels' in features[0] else None
        input_values = [f['input_values'] for f in features]

        batch = self.processor(input_values, padding=True, return_tensors='pt', sampling_rate=self.sampling_rate)

        max_input_length = batch['input_values'].shape[1]
        max_num_frames = max_input_length // self.samples_per_frame

        frame_masks = []
        for f in features:
            sample_start = f.get('sample_start', 0)
            sample_stop = f.get('sample_stop', max_input_length)

            start_frame = max(sample_start // self.samples_per_frame, 0)
            stop_frame = min((sample_stop + self.samples_per_frame - 1) // self.samples_per_frame, max_num_frames)

            mask = torch.zeros(max_num_frames, dtype=torch.float)
            mask[start_frame:stop_frame] = 1.0
            frame_masks.append(mask)

        frame_masks = torch.nn.utils.rnn.pad_sequence(frame_masks, batch_first=True, padding_value=0.0)

        batch['frame_mask'] = frame_masks

        if labels is not None:
            batch['labels'] = torch.tensor(labels)
        
        return batch






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

    model = Wav2Vec2WithAttentionClassifier.from_pretrained(
        '../wav2vec2-base',
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
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=masked_targets['train'],
        eval_dataset=masked_targets['test'],
        processing_class=feature_extractor,
        data_collator=data_collator
    )

    trainer.train()

    # tokenizer.save_pretrained(f'{model_dir}/{model}')
    # trainer.save_model(f'{model_dir}/{model}')