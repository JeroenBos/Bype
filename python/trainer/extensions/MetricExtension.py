from tensorflow.keras.models import Model  # noqa
from keyboard._3_scoring import Metrics, ValidationData
from trainer.trainer import TrainerExtension


class ValidationDataScoringExtensions(TrainerExtension): 

    def after_compile(self, model: Model) -> None:
        self.params.fit_args.callbacks.append(Metrics(self.params.validation_data, self.params.log_dir, model))
