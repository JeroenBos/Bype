from tensorflow.keras.models import Model  # noqa
from keyboard._3_scoring import Metrics, ValidationData
from trainer._trainer import TrainerExtension


class ValidationDataScoringExtensions(TrainerExtension): 

    def __init__(self, params):
        self.params = params

    def after_compile(self, model: Model) -> None:
        self.params.fit_args.callbacks.append(Metrics(self.params.validation_data, self.params.log_dir, model))
