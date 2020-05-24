from typing import Optional
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.callbacks import Callback  # noqa
from keyboard._3_scoring import Metrics, ValidationData
from trainer._trainer import TrainerExtension
from keyboard._0_types import SwipeEmbeddingDataFrame
from trainer.ModelAdapter import FitArgs
from typeguard import check_argument_types


class Params:
    validation_data: SwipeEmbeddingDataFrame
    fit_args: FitArgs



class ContinuousEpochCountExtensions(TrainerExtension):

    def __init__(self, params: Params, initial_initial_epoch=0):
        assert check_argument_types()
        super().__init__()
        self.params = params
        self._initial_initial_epoch = initial_initial_epoch


    def _state_name(self): 
        return 'initial_epoch_count'


    def initialize(self, prev_params: Optional[Params]):
        self.params.fit_args.callbacks.append(SetEpochIndexCallback(self.params))

        if prev_params is None:
            initial_epoch = self._initial_initial_epoch
        else:
            last_epoch = getattr(prev_params, self._state_name)
            initial_epoch = last_epoch + prev_params.i_epoch

        setattr(self.params, self._state_name, initial_epoch)
        self.params.fit_args.initial_epoch = initial_epoch


    def after_compile(self, model: Model) -> None:
        self.params.fit_args.callbacks.append(Metrics(
            validation_data=self.params.validation_data,
            log_dir=self.params.log_dir,
        ))


class SetEpochIndexCallback(Callback):
    def __init__(self, params):
        self._params = params

    def on_epoch_end(self, batch, logs={}):
        self._params.i_epoch += 1
