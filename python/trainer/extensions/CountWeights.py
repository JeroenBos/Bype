from typing import Any, Optional, Tuple, Union
import numpy as np
from tensorflow.keras.models import load_model, Model   # noqa
from trainer.ModelAdapter import FitArgs
from trainer.trainer import TrainerExtension
from trainer.types import X, Y
from utilities import memoize

def count_params(model: Union[Model, str]) -> int:
    if isinstance(model, str):
        model = load_model(model)
    """ This gets the _total_ number of parameters in the model, not necessarily all are trainable. """
    assert isinstance(model, Model)

    return model.count_params()


def count_params__memoized(model: Model) -> int:
    """ This gets the _total_ number of parameters in the model, not necessarily all are trainable. """
    assert isinstance(model, Model)

    return _count_params(_Hashable(model))


class _Hashable:
    def __init__(self, obj: Any):
        self.__obj = obj

    def __getattr__(self, name):
        return getattr(self.__obj, name)

    def __hash__(self):
        return id(self.model)



@memoize
def _count_params(model: _Hashable):
    return model.count_params()

class PrintWeightsCount(TrainerExtension):
    def cleanUp(self):
        try:
            print(count_params(self.params.best_model_path))
        except:  # noqa
            print("Failed to get weight count")
        return super().cleanUp()
