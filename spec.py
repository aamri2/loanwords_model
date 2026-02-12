"""
Provides the Spec family of classes, which provide all the relevant specifications for processing.

Also provides some convenience methods for automatically parsing specifications from strings.
"""

from typing import Sequence, Self, Container, NamedTuple, TypeVar, cast, Generic, Mapping
from collections.abc import Collection
from abc import ABC, abstractmethod, ABCMeta
from collections import defaultdict, namedtuple
import re
from functools import cache

_SEPARATOR = '_' # separator for string representations
T = TypeVar('T')

class SpecMeta(ABCMeta):
    """Metaclass for Spec."""

    name: str

    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.__name__)

class Spec(ABC, Generic[T], metaclass=SpecMeta):
    """
    ABC for specification-related classes.
    """

    name: str

    @classmethod
    @abstractmethod
    def allows(cls, value) -> bool:
        """Does the specification allow it?"""
    
    @property
    @abstractmethod
    def value(self) -> T:
        ...

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Value must be allowed by the specification."""
        
        if not self.allows(self.value):
            raise ValueError(f"Value {self.value} not allowed by {type(self)}.")
    
    @classmethod
    def cast_self(cls, value: T | Self) -> T:
        """Allows directly inputting an instance instead of its value."""

        if isinstance(value, cls):
            return cast(T, value.value)
        else:
            return cast(T, value) # type hinting does not detect that cls is Self
    
    @abstractmethod
    def __hash__(self, *args) -> int:
        """Implementations should pass values for instances."""

        return hash((self.name, *args))
    
    @abstractmethod
    def __str__(self, *args):
        """Implementations should pass values as string arguments."""
        
        return _SEPARATOR.join(args)

class SpecUnit(Spec[str]):
    """
    Inherited classes allow a simple string value from a constant set of values. Values may not contain the separator.
    """
    
    def __init__(self, value: str | Self, *args, **kwargs):
        if isinstance(value, str) and _SEPARATOR in value:
            raise ValueError(f"Values cannot contain the separator {_SEPARATOR}.")
        self._value = self.cast_self(value)
        super().__init__(value, *args, **kwargs)
    
    @property
    def value(self) -> str:
        return self._value
    
    @classmethod
    def allows(cls: type[Self], value: str | Self):
        return cls.cast_self(value) in cls._allowed_values

    def __init_subclass__(cls, name: str, allowed_values: Container[str]) -> None:
        cls._allowed_values = allowed_values
        cls.name = name

    def __hash__(self) -> int:
        return super().__hash__(self._value)
    
    def __str__(self):
        return super().__str__(self._value)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.value)})"

class SpecComplex(Spec[Sequence[Spec | Sequence[Spec] | None]]):
    """
    A specification containing multiple, ordered, named values that are themselves specifications.
    """
        
    def __init__(self, value: Self | Sequence[Spec | Sequence[Spec] | None] | str | None = None, *args, **kwargs):
        """
        Classes can be initialized with either an instance of itself, a sequence of values, or each value specified as keyword arguments.
        """

        if value:
            if isinstance(value, str):
                try:
                    self.__init__(self._from_str(value))
                    return
                except ValueError as e:
                    if not _SEPARATOR in value:
                        raise ValueError(f"String values must be in the special format using the separator {_SEPARATOR}.")
                    else:
                        raise e
            elif isinstance(value, (type(self), Sequence)):
                value_names = [value_type.name for value_type in self._value_types]
                value = self.cast_self(value)
                self.__init__(value = None, **{value_name: subvalue for value_name, subvalue in zip(value_names, value)})
                return
            else:
                raise TypeError(f"Positional argument must be an instance of {self.__class__.__name__} or a sequence of subvalues.")
        elif kwargs:
            for value_type in self._value_types:
                if not value_type.name in kwargs or not kwargs[value_type.name]:
                    setattr(self, value_type.name, None)
                else: # either instance of value_type, string, or sequence to cast to tuple of value_types
                    subvalue = kwargs[value_type.name]
                    if isinstance(subvalue, (value_type, str)):
                        setattr(self, value_type.name, value_type(subvalue))
                    elif isinstance(subvalue, Sequence):
                        subvalue = tuple(value_type(subvalue_i) for subvalue_i in subvalue)
                        setattr(self, value_type.name, subvalue)
                    else:
                        raise TypeError(f"{value_type} must be an instance of {value_type.__class__.__name__} or a string, or a sequence thereof.")
        else:
            raise ValueError("Must pass either an argument 'values' or keyword arguments for each value")

        super().__init__(self, *args, values = value, **kwargs)
    
    @property
    @cache
    def value(self) -> NamedTuple:
        """Returns a namedtuple of all values."""

        return self.ValueTuple(*[getattr(self, value_type.name) for value_type in self._value_types])

    @classmethod
    def allows(cls, value) -> bool:
        if isinstance(value, Sequence):
            if len(value) != len(cls._value_types):
                return False
            else:
                for subvalue, value_type in zip(value, cls._value_types): # check for multiple, optional, and allowed
                    if not subvalue:
                        if not value_type in cls._optional:
                            return False
                    elif isinstance(subvalue, Sequence) and not isinstance(subvalue, str):
                        if value_type not in cls._multiple:
                            return False
                        if any(not value_type.allows(value_type.cast_self(subvalue_i)) for subvalue_i in subvalue):
                            return False
                    elif not isinstance(subvalue, value_type) and not value_type.allows(subvalue):
                        return False
                return True
        else:
            return False

    def __init_subclass__(cls, name: str, value_types: Sequence[type[Spec]], optional: Collection[str | type[Spec]] | str | type[Spec] | None = None, multiple: Collection[str | type[Spec]] | str | type[Spec] | None = None):
        """
        Args:
            value_types:
                A sequence of unique value types, in order.
            optional:
                Specifies a value or some values that can receive a value
                of None. Can be specified as strings or as classes.
            multiple:
                Specifies a value or some values that can receive multiple values.
        """

        if not value_types:
            raise ValueError("Must provide at least one value type.")
        
        cls.name = name
        value_names = {value_type.name: value_type for value_type in value_types}

        if optional:
            if isinstance(optional, str):
                if optional not in value_names:
                    raise ValueError(f"Optional value {optional} not in values {list(value_types)}.")
                optional = {value_names[optional]}
            elif isinstance(optional, type) and issubclass(optional, Spec):
                if optional not in value_types:
                    raise ValueError(f"Optional value {optional} not in values {list(value_types)}.")
                optional = {optional}
            elif isinstance(optional, Collection):
                if not set(optional) < set(value_types):
                    raise ValueError("Optional values must only include some (but not all) values.")
                else:
                    optional = set(value_names[str(optional_i)] for optional_i in optional)
            else:
                raise TypeError("Optional values must be a string or Spec, or a collection thereof.")
        else:
            optional = frozenset()
        if multiple:
            if isinstance(multiple, str):
                if multiple not in value_names:
                    raise ValueError(f"Multiple value {multiple} not in values {list(value_types)}.")
                multiple = {value_names[multiple]}
            elif isinstance(multiple, type) and issubclass(multiple, Spec):
                if multiple not in value_types:
                    raise ValueError(f"Multiple value {multiple} not in values {list(value_types)}.")
                multiple = {multiple}
            elif isinstance(multiple, Collection):
                if not set(multiple) <= set(value_types):
                    raise ValueError(f"Multiple values {set(multiple) - set(value_type.name for value_type in value_types)} not in values {list(value_type.name for value_type in value_types)}.")
                else:
                    multiple = set(value_names[str(multiple_i)] for multiple_i in multiple)
            else:
                raise TypeError("Multiple values must be a string or Spec, or a collection thereof.")
        else:
            multiple = frozenset()
        cls._optional = frozenset(optional)
        cls._multiple = frozenset(multiple)

        for value_type in value_types:
            if not issubclass(value_type, Spec):
                raise TypeError("Value types must be subclasses of the Spec class.")
            if hasattr(cls, value_type.name):
                raise ValueError(f"Value type {value_type} conflicts with a class attribute.")
        cls._value_types = tuple(value_types)
    
        cls.ValueTuple = namedtuple('ValueTuple', [value_type.name for value_type in cls._value_types])
    
    def __hash__(self):
        return super().__hash__(*(getattr(self, value_type.name) for value_type in self._value_types))
    
    def __str__(self):
        parts = []

        for value_type in self._value_types:
            if getattr(self, value_type.name):
                subvalue = getattr(self, value_type.name)
                if isinstance(subvalue, Sequence):
                    parts.extend(subvalue_i.__str__() for subvalue_i in subvalue)
                else:
                    parts.append(subvalue.__str__())
        return super().__str__(*parts)
    
    def __repr__(self):
        parameters = [f"{value_type} = {repr(getattr(self, value_type.name))}" for value_type in self._value_types]
        return f"{self.__class__.__name__}({', '.join(parameters)})"
    
    @classmethod
    def _from_str(cls, specification: str) -> Self:
        """
        Parses string representations of the form generated by __str__. Expects SpecUnits and SpecComplexes.
        
        Assumes defaultdict respects order.
        """

        parts = specification.split(_SEPARATOR)
        value_types = list(cls._value_types)
        value: defaultdict[str, list[Spec | str] | Spec | str | None] = defaultdict(list)

        while parts and value_types: # greedily decode; remove as matches are found
            value_type = value_types.pop(0)

            if issubclass(value_type, SpecUnit): # only a single part necessary
                if not value_type.allows(parts[0]):
                    if value_type in cls._optional:
                        value[value_type.name] = None
                        continue
                    else:
                        raise ValueError(f"Attempted to assign invalid value {parts.pop(0)} to {value_type}.")
                else:
                    if value_type in cls._multiple: # append until failure
                        while parts and value_type.allows(parts[0]):
                            cast(list, value[value_type.name]).append(parts.pop(0)) # type checking can't tell this must be a list
                    else:
                        value[value_type.name] = parts.pop(0)
            elif issubclass(value_type, SpecComplex): # try shorter parts until one succeeds; pseudo-recursive
                done = False
                while not done:
                    for i in reversed(range(len(parts))):
                        try:
                            cast(list, value[value_type.name]).append(value_type._from_str(_SEPARATOR.join(parts[:i + 1]))) # type checking can't tell this must be a list
                            parts = parts[i + 1:]
                            if value_type not in cls._multiple:
                                done = True
                            break
                        except (ValueError, NotImplementedError):
                            continue
                    else: # could not find parsing
                        if value[value_type.name]: # may have found in earlier loop
                            done = True
                            break
                        elif value_type in cls._optional:
                            value[value_type.name] = None
                            done = True
                            break
                        else:
                            raise ValueError(f"Could not find valid {value_type} from {_SEPARATOR.join(parts)}.")
            else:
                raise NotImplementedError("Cannot decode SpecComplex from string with non-SpecUnit/SpecComplex values.")
        else:
            value_types = [value_type for value_type in value_types if value_type not in cls._optional]
            if parts or value_types: # unused parts or unfilled mandatory values
                raise NotImplementedError("Haven't figured out handling weird cases yet.")
            # change lists of 1 to singletons
            for value_name, subvalue in value.items():
                if isinstance(subvalue, list) and len(subvalue) == 1:
                    value[value_name] = subvalue[0]
            return cls(value = None, **value) # typing complains if value not specified

class BaseSpec(SpecUnit, name = 'base', allowed_values = ['w2v2', 'w2v2fr']):
    """Specification unit for base models."""

    @property
    def architecture(self) -> str:
        if self.value in ['w2v2', 'w2v2fr']:
            return 'Wav2Vec2'
        else:
            raise NotImplementedError(f'Architecture for base model {self.value} unspecified.')

class LayerSpec(SpecUnit, name = 'layer', allowed_values = ['attn', 'max', 'tempmax', 'hiddenmax', 'relu', 'class', 'transformer', 'ctc', 'mean']):
    """Specification unit for layer types."""

class NaturalNumbers(Container):
    def __contains__(self, x: str) -> bool:
        return x.isdigit() and int(x) >= 0

class FrozenSpec(SpecUnit, name = 'frozen', allowed_values = NaturalNumbers()):
    """Number of layers frozen."""

    def __init__(self, value: str | int | Self, *args, **kwargs):
        # cast int to string
        if isinstance(value, int):
            value = str(value)
        super().__init__(value, *args, **kwargs)

class TrainingDatasets(Container):
    base_datasets = ['timit', 'librispeech', 'librispeechFR', 'bl', 'wvEN', 'wvResponses', 'wvENResponses', 'wvENResponses10Fold']
    neutral_variants = ['EV', 'MV', 'S', 'A', 'CL', 'N'] # these don't affect parsing in any way
    parser = re.compile(f"({'|'.join(base_datasets)})({'|'.join(neutral_variants)})*") # base_dataset + optional neutral_variant

    def __contains__(self, x: str) -> bool:
        return bool(self.parser.fullmatch(x))
    
    def family(self, x: str) -> str:
        if x in self:
            return cast(re.Match[str], self.parser.search(x)).group()
        else:
            raise ValueError(f"{x} is not a valid dataset.")

class TrainingDatasetSpec(SpecUnit, name = 'training_dataset', allowed_values = TrainingDatasets()):
    """Specification for training datasets."""

    _allowed_values: TrainingDatasets

    @property
    def family(self):
        return self._allowed_values.family(self.value)

class TrainingVars(Container):
    value: str
    allowed_patterns = ['v[0-9]+', '[0-9]+e', '[0-9]+', 'best', 'cross', 'N', 'var[a-z|A-Z]+'] # regex patterns; versions, epoch counts, initializations
    parser = re.compile(f"({'|'.join(allowed_patterns)})")

    def __contains__(self, x: str) -> bool:
        return bool(self.parser.fullmatch(x))

class TrainingVarSpec(SpecUnit, name = 'training_var', allowed_values = TrainingVars()):
    """Variants of training that do not affect processing."""

    value: TrainingVars

class TrainingSpec(SpecComplex, name = 'training', value_types = [LayerSpec, FrozenSpec, TrainingDatasetSpec, TrainingVarSpec], optional = TrainingVarSpec, multiple = [LayerSpec, TrainingVarSpec]):
    """Specification for model trainings."""

    layer: LayerSpec | tuple[LayerSpec, ...]
    frozen: FrozenSpec
    training_dataset: TrainingDatasetSpec
    training_var: TrainingVarSpec | None | tuple[TrainingVarSpec, ...]

class PoolingMethodSpec(SpecUnit, name = 'method', allowed_values = ['centreMeans', 'cvc', 'decode']):
    """Specification unit for pooling method, if model does not directly output classifications."""
    
    @property
    def possible_targets(self) -> Container[str]:
        if self.value in ['centreMeans', 'cvc']:
            return ['vowels']
        else:
            return ['vowels', 'cvc']

class PoolingTargetSpec(SpecUnit, name = 'target', allowed_values = ['vowels', 'cvc']):
    """Specification unit for pooling target."""

class PoolingSpec(SpecComplex, name = 'pooling', value_types = [PoolingMethodSpec, PoolingTargetSpec]):

    method: PoolingMethodSpec
    target: PoolingTargetSpec

    @classmethod
    def allows(cls, value) -> bool:
        if super().allows(value): # also must be a valid target for the method
            value = cls.ValueTuple(*value)
            target: PoolingTargetSpec = value.target
            method: PoolingMethodSpec = value.method
            return target.value in method.possible_targets
        else:
            return False

class ModelSpec(SpecComplex, name = 'model', value_types = [BaseSpec, TrainingSpec], multiple = TrainingSpec):
    """Specification for models."""

    base: BaseSpec
    training: TrainingSpec | tuple[TrainingSpec, ...]
    
    @property
    def needs_pooling(self) -> bool:
        """Does the specified model need pooling?"""

        return self.output_layer.value != 'class'

    @property
    def layers(self) -> tuple[BaseSpec, *tuple[LayerSpec, ...]]:
        """Returns the base model and additional layers from training."""

        _layers: list[LayerSpec] = []

        if isinstance(self.training, Sequence):
            training = self.training
        else:
            training = [self.training]

        for training_i in training:
            if isinstance(training_i.layer, Sequence):
                _layers.extend(list(training_i.layer))
            else:
                _layers.append(training_i.layer)

        return (self.base, *_layers)

    @property
    def output_layer(self) -> LayerSpec:
        """The specification of the final layer in the model."""

        return self.layers[-1]
    
    @property
    def output_dataset(self) -> TrainingDatasetSpec:
        """The specification of the final training dataset in the model."""
        
        if isinstance(self.training, Sequence):
            _output_dataset = self.training[-1].training_dataset
        else:
            _output_dataset = self.training.training_dataset

        if isinstance(_output_dataset, Sequence):
            return _output_dataset[-1]
        else:
            return _output_dataset

class HumanSpec(SpecUnit, name = 'humans', allowed_values = ['humans', 'humansFR']):
    """Specification of participant response pool."""

class TestDatasetSpec(SpecUnit, name = 'test_dataset', allowed_values = ['wv']):
    """Specification of dataset containing test stimuli."""

class ProbabilitySpec(SpecComplex, name = 'probability', value_types = [ModelSpec, PoolingSpec, TestDatasetSpec], optional = PoolingSpec):
    """Specification of probabilities for models."""
    
    model: ModelSpec
    pooling: PoolingSpec | None
    test_dataset: TestDatasetSpec

    @classmethod
    def allows(cls, value) -> bool:
        if super().allows(value):
            value = cls.ValueTuple(*value)
            model: ModelSpec = value.model
            pooling: PoolingSpec | None = value.pooling
            return not (model.needs_pooling and not pooling)
        else:
            return False
    
class HumanProbabilitySpec(SpecComplex, name = 'human_probability', value_types = [HumanSpec, TestDatasetSpec]):
    """Specification of probabilities for humans."""

    humans: HumanSpec
    test_dataset: TestDatasetSpec