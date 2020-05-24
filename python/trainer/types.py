from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, TypeVar, Tuple, Dict, Iterable, List, Union, Optional
import numpy as np
from tensorflow.keras import Model  # noqa
from functools import reduce
from utilities import memoize, override, virtual, sealed, abstract
from keyboard._0_types import SwipeEmbeddingDataFrame
from tensorflow.keras.callbacks import History  # noqa

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
    def create_model(self, model: Optional[IModel]) -> Optional[IModel]:
        return model

    def before_compile(self, model: IModel) -> None:
        pass

    def after_compile(self, model: IModel) -> None:
        pass

    def before_fit(self, x: X, y: Y) -> Tuple[X, Y]:
        return x, y

    def after_fit(self, history: History, x: X, y: Y) -> None:
        pass
