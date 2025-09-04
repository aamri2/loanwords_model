# Contains custom classes for building models

import torch
from torch import nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Union, Any
from dataclasses import dataclass

class AttentionPooling(nn.Module):
    """
    Scores each item in a sequence, and then returns a weighted mean.
    """

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
    
class Wav2Vec2ForCTCWithAttentionClassifier(Wav2Vec2ForCTC):
    """
    Pretrained Wav2Vec2ForCTC with an additional classifying head.
    """

    def __init__(self, config):
        super().__init__(config)
        self.attention_pooling = AttentionPooling(config.vocab_size)
        self.classifier = nn.Linear(config.vocab_size, config.num_labels)
        self.init_weights()

    def freeze_base_model(self): # freeze everything but the classification head
        """Freezes Wav2Vec2 and CTC layers, so only the attention and classifier get trained."""

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        for param in self.dropout.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def forward(self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        frame_mask = None,
    ):

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.lm_head(hidden_states)

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
    
_HIDDEN_STATES_START_POSITION = 2
class Wav2Vec2ForCTCWithTransformer(Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2ForCTC with a transformer encoder before the CTC layer.
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1
        if self.config.use_weighted_layer_sum:
            self.hidden_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=config.output_hidden_size, nhead=8, dim_feedforward=config.output_hidden_size, batch_first=True),
            num_layers=6
        )
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self): # freeze wav2vec2
        """Freezes Wav2Vec2 and CTC layers, so only the attention and classifier get trained."""

        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states # need hidden states for this

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.hidden_layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        
        hidden_states = self.transformer(hidden_states)
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class Wav2Vec2ForCTCWithTransformerL2(Wav2Vec2ForCTCWithTransformer):
    """A model that allows training CTC on top of another CTC model."""

    def __init__(self, config, l2_vocab_size):
        super().__init__(config)
        self.l2_head = nn.Linear(config.vocab_size, l2_vocab_size)
        self.post_init()

    def freeze_l1_model(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.dropout.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states # need hidden states for this

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.hidden_layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        
        hidden_states = self.transformer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.lm_head(hidden_states)
        logits = self.l2_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

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

    def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
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