from dataclasses import dataclass, field
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, TypeVar, Tuple, Dict, Iterable, List, Union, Optional
import numpy as np
from tensorflow.keras import Model  # noqa
from functools import reduce
from utilities import memoize, override, virtual, sealed, abstract
from trainer.types import IModel, History, X, Y, TrainerExtension
from typeguard import check_argument_types, check_return_value  # noqa


class IModelAdapter(ABC):
    @abstractmethod
    def fit(self, model: IModel, x: X, y: Y) -> History:
        raise ABC

    @abstractmethod
    def compile(self, model: IModel) -> None:
        raise ABC

    @sealed
    def concretify(self, model: IModel) -> IModel:
        adapter = self

        class AdapterModel(IModel):
            def compile(self):
                return adapter.compile(model)

            def fit(self, x, y):
                return adapter.fit(model, x, y)
        return AdapterModel()



class ModelAdapter(IModelAdapter, ABC):
    """ 
    This class is reponsible for applying arguments to a model.
    """

    @abstractmethod
    def get_fit_args(self) -> Dict:
        return dict()

    @abstractmethod
    def get_compile_args(self) -> Dict:
        return dict()

    def fit(self, model: Model, x: X, y: Y) -> History:
        fit_kwargs = self.get_fit_args()
        return model.fit(x, y, **fit_kwargs)

    def compile(self, model: Model) -> None:
        compile_args = self.get_compile_args()
        return model.compile(**compile_args)


class ArgsAdapter:
    def to_dict(self) -> Dict:
        result = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return result



class Params:
    fit_args: ArgsAdapter
    compile_args: ArgsAdapter



class ParameterizedModelAdapter(ModelAdapter):
    """ 
    This class is reponsible for extracting arguments from `TParams` and applying them to a model.
    """

    def __init__(self, params: Params):
        assert check_argument_types()
        self._fit_args = params.fit_args.to_dict()
        self._compile_args = params.compile_args.to_dict()

    @sealed
    @override
    def get_fit_args(self) -> Dict:
        result = {**super().get_fit_args(), **self._fit_args}
        return result

    @sealed
    @override
    def get_compile_args(self) -> Dict:
        return {**super().get_compile_args(), **self._compile_args}


class ParameterizeModelExtension(TrainerExtension):
    """ Ensures that `params.fit_args` and `params.compile_args` are passed to the `model.fit` and `model.compile` calls. """

    def __init__(self, params: Params):
        check_argument_types()
        self._params = params

    @sealed
    def create_model(self, model: IModel) -> IModel:
        assert model is not None, "An extension creating a model must be provide before this one"

        adapter = ParameterizedModelAdapter(self._params)
        return adapter.concretify(model)


class ParameterizedCreateModelBase(ParameterizeModelExtension, ABC):
    """ Ensures that `params.fit_args` and `params.compile_args` are passed to the `model.fit` and `model.compile` calls. 
    Assumes the extension creating the model inherits from this extension. 
    """

    def __init__(self, params):
        self._params = params

    def create_model(self, model: Optional[IModel]) -> IModel:
        assert model is None, "Another extension already created a model"
        model = self._create_model()
        return super().create_model(model)

    @abstractmethod
    def _create_model(self) -> IModel:
        raise ABC

@dataclass
class FitArgs(ArgsAdapter):
    epochs: int
    callbacks: List = field(default_factory=list)

@dataclass
class CompileArgs(ArgsAdapter):
    loss: str = 'mean_squared_error'
    optimizer: str = 'adam'
    metrics: List = field(default_factory=lambda: ['accuracy'])
