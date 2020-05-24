from abc import ABC, abstractproperty, abstractmethod
from trainer.trainer import TrainerExtension
from trainer.types import IModel, History, X, Y, Tuple
from typing import Any, Optional
from utilities import virtual, sealed


class ComputeValueTrainerExtension(TrainerExtension, ABC):
    def __init__(self, params: Any, prev_params: Any):
        self.params = params
        self.prev_params = prev_params

        if self.compute_on_init:
            self._invoke()

    def _invoke(self):
        _param_name = self.param_name
        assert isinstance(_param_name, str), "param_name must return a string"
        if not hasattr(self.params, _param_name) or getattr(self.params, _param_name) is None:
            value = self.compute()
            setattr(self.params, _param_name, value)

    @abstractproperty
    def param_name(self) -> str:
        raise ABC

    @abstractmethod
    def compute(self) -> Any:
        raise ABC

    @virtual
    @property
    def compute_on_init(self) -> bool:
        return True

    @virtual
    @property
    def compute_on_before_fit(self) -> bool:
        return False

    @virtual
    @property
    def compute_on_before_compile(self) -> bool:
        return False

    @sealed
    def before_compile(self, model: IModel) -> None:
        if self.compute_on_before_compile:
            self._invoke(model)

    @sealed
    def before_fit(self, x: X, y: Y) -> Tuple[X, Y]:
        if self.compute_on_before_fit:
            self._invoke(x, y)
        return x, y

    @property
    def current_value(self):
        _param_name = self.param_name
        assert isinstance(_param_name, str), "param_name must return a string"

        return getattr(self.params, _param_name, None)

    @property
    def prev_value(self):
        _param_name = self.param_name
        assert isinstance(_param_name, str), "param_name must return a string"

        if self.prev_params is None:
            return None
        return getattr(self.prev_params, _param_name, None)
