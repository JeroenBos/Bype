from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, TypeVar, Tuple, Dict, Iterable, List, Union, Optional
import numpy as np
from tensorflow.keras import Model  # noqa
from functools import reduce
from utilities import memoize, override, virtual, sealed, abstract
from keyboard._0_types import SwipeEmbeddingDataFrame
from tensorflow.keras.callbacks import History  # noqa
import trainer


TParams = TypeVar('TParams')
X, Y = TypeVar('X'), TypeVar('Y')


class IModelAdapter(ABC):
    @abstractmethod
    def fit(self, model: "IModel", x: X, y: Y) -> History:
        raise ABC

    @abstractmethod
    def compile(self, model: "IModel") -> None:
        raise ABC


IModel = Union[Model, IModelAdapter]

class TrainerExtension:
    @sealed
    @property
    def params(self) -> TParams:
        """
        Gets the current params. Only available after `__init__`.
        """

        assert hasattr(self, '_params'), "The params cannot be used until after `__init__`"

        return self._params

    @sealed
    @property
    def prev_params(self) -> Optional[TParams]:
        """
        Gets the params in the previous round; or None if this is the first round.
        """

        assert hasattr(self, '_prev_params'), "The prev_params cannot be used until after `__init__`"

        return self._prev_params

    @sealed
    @property
    def is_first_stage(self) -> bool:
        return self.prev_params is None

    @virtual
    def initialize(self) -> None:
        pass

    @virtual
    def create_model(self, model: Optional[IModel]) -> Optional[IModel]:
        return model

    @virtual
    def before_compile(self, model: IModel) -> None:
        pass

    @virtual
    def after_compile(self, model: IModel) -> None:
        pass

    @virtual
    def before_fit(self, x: X, y: Y) -> Tuple[X, Y]:
        return x, y

    @virtual
    def after_fit(self, history: History, x: X, y: Y) -> None:
        pass
