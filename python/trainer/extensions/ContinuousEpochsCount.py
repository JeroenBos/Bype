from typing import Any, Optional
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.callbacks import Callback  # noqa
from keyboard._3_scoring import Metrics, ValidationData
from trainer.trainer import TrainerExtension
from keyboard._0_types import SwipeEmbeddingDataFrame
from trainer.ModelAdapter import FitArgs
from trainer.extensions.ComputeValueExtension import ComputeValueTrainerExtension
from utilities import read_all, overwrite_all, override

continued_epoch_file_name = 'epoch_count.txt'

class Params:
    fit_args: FitArgs
    filebased_continued_epoch_counting: Optional[bool]
    log_dir: str



class ContinuousEpochCountExtensions(ComputeValueTrainerExtension):

    def __init__(self, params: Params, prev_params: Optional[Params]):
        super().__init__(params, prev_params)
        self.params.fit_args.callbacks.append(SetEpochIndexCallback(params))

        # totally optional:
        self.params.initial_epoch_count = self.prev_value or 0   # prev_value is prev_epoch_count

    @property
    def path(self) -> str:
        return self.params.log_dir + continued_epoch_file_name


    @property
    def param_name(self) -> str:
        return 'epoch_count'

    def compute(self) -> Any:
        # Does not get called when epoch_count is already set
        if self.prev_params is None:
            if getattr(self.params, 'filebased_continued_epoch_counting', False):
                try:
                    return int(read_all(self.path))
                except:  # noqa
                    pass
            return 0

        return self.prev_value + 1

    def after_fit(self, history, x, y):
        overwrite_all(self.path, str(self.current_value))
        return super().after_fit(history, x, y)




class SetEpochIndexCallback(Callback):
    def __init__(self, params):
        self._params = params

    def on_epoch_end(self, batch, logs={}):
        self._params.epoch_count += 1


class ApplyInitialEpochAndNumEpochToFitArgsTrainerExtension(TrainerExtension):

    def __init__(self, params):
        self.params = params

    def before_fit(self, x, y):
        initial_epoch = getattr(self.params, 'epoch_count', 0)
        n_epochs = getattr(self.params, 'n_epochs')
        assert n_epochs is not None, "'n_epochs' is mandatory"

        self.params.fit_args.initial_epoch = initial_epoch
        self.params.fit_args.epochs = initial_epoch + n_epochs

        return x, y
