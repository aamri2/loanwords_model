# Documentation

## Experiment

The full list of model + probability specifications to compare is:
- `w2v2_`
    - `ctc_`
        - `0_timit_`
            - `centreMeans_vowels`
            - `cvc_vowels`
        - `1_timit_`
            - `centreMeans_vowels` ✅
            - `cvc_vowels` ✅
        - `2_timit_`
            - `centreMeans_vowels`
            - `cvc_vowels`
- `w2v2fr_`

See also [abandoned branches](#archived-experiment).

Existing probabilities are marked ✅

## Naming conventions

### Probabilities

Probabilities are named according to the following convention:

> `p_model(_poolingMethod)(_outputs)_dataset`

where
- `p`: The letter 'p' (for 'probabilities')
- `model`: The full model specification. See [naming conventions](#models). `human` is used for human responses.
- `poolingMethod`: If the model does not put out probabilities that can be used directly (e.g. a CTC model), this describes the method used to get the probabilities. The methods in use are:
    - `centreMean`: Take the mean of the three centremost frames of the output sequence.
    - `cvc`: Use a special beam search algorithm that only looks for CVC sequences, where consonants are pooled.
- `outputs`: The possible outputs that the probabilities select between, if not all outputs from the model are used. Right now, the only outputs in use are:
    - `vowels`: English vowel classifications
- `dataset`: The dataset used as stimuli. The only dataset in use for this is `wv`.

#### Examples

### Models

Models (and related files) are named according to the following convention:

> `m_pretrainedModel_heads_N_dataset`

where
- `m`: The letter 'm' (for 'model')
- `pretrainedModel`: the model used for pretraining. See [naming conventions](#pre-trained-models).
- `heads`: One or more heads added to the model, separated by underscores. Options include:
    - `ctc`: For a CTC head.
    - `class`: For a linear classification head.
    - `attn`: For an attention pooling layer (made with `AttentionPooling`).
- `N`: the number of frozen layers
    - for Wav2Vec2-based models, `0` is full fine-tuning, `1` is a frozen feature encoder (using `freeze_feature_encoder()`), `2` is a fully frozen Wav2Vec2 model (using `freeze_base_model()`), and any higher number corresponds to each classification head.
- `dataset`: the dataset on which the model was fine-tuned (including modified datasets). See [naming conventions](#datasets).

The full string, excluding the initial `m_`, is the 'model specification', and should be sufficient to distinguish two models.

#### Examples

A model consisting of a wav2vec2-base-fr-v2 model with a CTC head fine-tuned on the BL-database (Blue Lips), where the feature encoder was frozen, would be called `m_w2v2fr_ctc_1_bl`.

A model that then adds a classification layer (with attention pooling) on top of this, using World Vowels stimuli, would be `m_w2v2fr_ctc_1_bl_attn_class_3_wv`. Here, `w2v2fr_ctc_1_bl` is treated as a complex pre-trained model of its own.

### Pre-trained models

The following pre-trained models are in use, with the following abbreviations:
- `w2v2`: [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- `w2v2fr`: [facebook/wav2vec2-base-fr-voxpopuli-v2](https://huggingface.co/facebook/wav2vec2-base-fr-voxpopuli-v2)

Any fine-tuned model can then be used as a pre-trained model for another head. In this case, use the full model specification as the name of the pre-trained model.

### Datasets

Possible prefixes:
- `prep_`: The dataset has been prepared for training.
- `pretrainedModel_`: The dataset has been run through the pretrained model.

The following datasets are in use, with the following abbreviations:
- `timit`: TIMIT
- `timitEV`: Extracted vowels from TIMIT
- `timitMV`: Masked vowels from TIMIT
- `bl`: BL-Database (Blue Lips)
- `wv`: World Vowels stimuli

## Appendix

### Archived experiment

- `w2v2_`
    - `ctc_`
        - `0_`
            - ~~`attn_class_3_`~~
                - ~~`timitEV`~~
                - ~~`timitMV`~~
                - ~~`wvEN`~~
        - `1_`
            - ~~`attn_class_3_`~~
                - ~~`timitEV`~~
                - ~~`timitMV`~~
                - ~~`wvEN` ✅~~
        - `2_`
            - ~~`attn_class_3_`~~
                - ~~`timitEV`~~
                - ~~`timitMV`~~
                - ~~`wvEN`~~
    - ~~`attn_class_`~~
        - ~~`0_`~~
            - ~~`timitEV`~~
            - ~~`timitMV`~~
        - ~~`1_`~~
            - ~~`timitEV`~~
            - ~~`timitMV`~~
        - ~~`2_`~~
            - ~~`timitEV` ✅~~
            - ~~`timitMV` ✅~~