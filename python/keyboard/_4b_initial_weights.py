from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model, Model  # noqa
from KerasModelPadder import copy_weights


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
    def __init__(self, path):
        self.model = load_model(path)

    def weights(self, model: Model) -> None:
        copy_weights(model, self.model)
