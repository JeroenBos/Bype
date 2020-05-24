# this is my attempt at a BaseEstimator like from sklearn, but then sane (like I said: "attempt")
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, TypeVar, Tuple, Dict, Iterable, List, Union, Optional
import numpy as np
from tensorflow.keras import Model  # noqa
from functools import reduce
from utilities import memoize, override, virtual, sealed, abstract
from trainer.types import IModel, X, Y, History, TrainerExtension

# Trainer and TrainerBase _could_ be used outside of this module, but probably shouldn't
class TrainerBase(ABC):
    """ Provides an API to interact with the trainer extensions as a whole. """
    @abstractmethod
    def get_extensions(self) -> Iterable[TrainerExtension]:
        pass

    def _initialize(self):
        for extension in self.get_extensions():
            extension.initialize(self)

    @sealed
    def create_model(self) -> IModel:
        model = None
        for extension in self.get_extensions():
            model = extension.create_model(model)
        assert model is not None, "None of the extensions created a model"
        return model

    @sealed
    def _before_compile(self, model: IModel) -> None:
        for extension in self.get_extensions():
            extension.before_compile(model)

    @sealed
    def compile(self, model: IModel) -> None:
        self._before_compile(model)
        none = model.compile()
        self._after_compile(model)
        return none

    @sealed
    def _after_compile(self, model: IModel) -> None:
        for extension in self.get_extensions():
            extension.after_compile(model)

    @sealed
    def _before_fit(self, x, y) -> Tuple[X, Y]:
        args = x, y
        for extension in self.get_extensions():
            args = extension.before_fit(*args)
        return args

    @sealed
    def fit(self, model: IModel, x: X, y: Y) -> History:
        x, y = self._before_fit(x, y)
        history = model.fit(x, y)
        self._after_fit(history, x, y)
        return history

    @sealed
    def _after_fit(self, history: History, x, y) -> None:
        for extension in self.get_extensions():
            extension.after_fit(history, x, y)

    def train(self, x: X, y: Y) -> History:
        model = self.create_model()
        self.compile(model)
        return self.fit(model, x, y)



class Trainer(TrainerBase):
    def __init__(self, extensions: List[TrainerExtension]):
        super().__init__()
        self._extensions = extensions

    @override
    def get_extensions(self):
        return self._extensions


class ParameterizedTrainer(TrainerBase):
    def __init__(self, extensions: List[TrainerExtension], params):
        super().__init__()
        self._extensions = extensions

    @override
    def get_extensions(self):
        return self._extensions


TParams = TypeVar('TParams')


class TrainingsPlanBase(ABC):
    @abstractproperty
    def params(self) -> Iterable[TParams]:
        raise ABC

    @abstractmethod
    def get_extensions(self, params: TParams, prev_params: Optional[TParams]) -> Iterable[TrainerExtension]:
        raise ABC

    def execute(self, data: Optional[Tuple[X, Y]] = None):
        """
        Executes the current trainings plan.

        :param data: If no trainings data is provided, it is assumed that one of the extensions will replace (None, None) with real data. 
        """
        data = data if data is not None else (None, None)
        prev_params = None
        for params in self.params:
            extensions = list(self.get_extensions(params, prev_params))

            trainer = Trainer(extensions)
            trainer.train(*data)

            prev_params = params
