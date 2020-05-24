from typing import Tuple, Optional

from trainer.types import TrainerExtension, IModel


class LoadInitialWeightsTrainerExtension(TrainerExtension):
    def __init__(self, params):
        self.params = params

    def create_model(self, model: Optional[IModel]) -> Optional[IModel]:

        return model
