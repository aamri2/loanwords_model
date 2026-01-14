"""Deals with wrapping models for pooling logits when necessary."""

from typing import Any
from spec import PoolingSpec
from model_handler import Model


class Pooling():
    """Transforms model logits to classifications as necessary."""

    spec: PoolingSpec

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass