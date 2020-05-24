from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model, Model  # noqa
from KerasModelPadder import copy_weights
import os

class WeightInitStrategy(ABC):
    no_init: "WeightInitStrategy"

    @abstractmethod
    def init_weights(self, model: Model) -> None:
        pass

class NoInitStrategy(WeightInitStrategy):
    def init_weights(self, model: Model) -> None:
        pass


WeightInitStrategy.no_init = NoInitStrategy()




class InitialWeights(WeightInitStrategy):
    """ Loads the h5 file once and applies it on every initialization. """

    def __init__(self, path):
        self.model = load_model(path)

    def init_weights(self, model: Model) -> None:
        copy_weights(model, self.model)


class ReloadWeights(WeightInitStrategy):
    """ Reloads the h5 file every initialization (so it may be updated concurrently). """

    def __init__(self, get_path):
        self.get_path = get_path

    def init_weights(self, model: Model) -> None:
        path = self.get_path(global_phase)
        if not os.path.exists(path):
            path = self.get_path(max(0, global_phase - 1))

        copy_weights(model, path) 
