from typing import Callable
from spec import BaseSpec
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2FeatureExtractor
from model_architecture import FormantFeatureExtractor

_BASE_PATH = '../'

class Base():
    spec: BaseSpec

    def __init__(self, spec: str | BaseSpec):
        self.spec = BaseSpec(spec)

    @property
    def path(self) -> str:
        return f"{_BASE_PATH}{self.spec}"
    
    @property
    def tokenizer_class(self) -> Callable[[str], PreTrainedTokenizer]:
        if self.spec.architecture == 'Wav2Vec2':
            def wrapped_tokenizer_class(vocab_file: str) -> PreTrainedTokenizer:
                return Wav2Vec2PhonemeCTCTokenizer(vocab_file, do_phonemize = False)
            return wrapped_tokenizer_class
        else:
            raise NotImplementedError(f"Tokenizer class unknown for architecture {self.spec.architecture}.")
        
    @property
    def feature_extractor(self):
        if self.spec.architecture in ['Wav2Vec2', 'MFCC']:
            return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        elif self.spec.architecture == 'formant':
            return FormantFeatureExtractor(sampling_rate=16000, feature_size=5)
        else:
            raise NotImplementedError(f"Feature extractor class unknown for architecture {self.spec.architecture}.")