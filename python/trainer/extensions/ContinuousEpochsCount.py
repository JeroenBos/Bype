from unordereddataclass import mydataclass
from typing import Any, Optional, Tuple
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.callbacks import Callback  # noqa
from keyboard._3_scoring import Metrics, ValidationData
from trainer.trainer import TrainerExtension
from keyboard._0_types import SwipeEmbeddingDataFrame
from trainer.ModelAdapter import FitArgs
from trainer.extensions.ComputeValueExtension import ComputeValueTrainerExtension
from utilities import read_all, overwrite_all, override

continued_epoch_file_name = 'epoch_count.txt'
continued_stage_file_name = 'stage_count.txt'

@mydataclass
class Params:
    # fit_args: FitArgs // duplication doesn't merge but create additional fields :S
    filebased_continued_epoch_counting: Optional[bool] = None
    filebased_continued_stage_counting: Optional[bool] = True
    log_dir: str
    # set initial_epoch_count


class ContinuousEpochCountExtensions(ComputeValueTrainerExtension):
    @override
    @property
    def params(self) -> Params:
        return super().params

    @override
    def initialize(self):
        super().initialize()
        self.params.fit_args.callbacks.append(SetEpochIndexCallback(self.params))

        # totally optional:
        self.params.initial_epoch_count = self.prev_value or 0   # prev_value is prev_epoch_count

    @property
    def path(self) -> str:
        return self.params.log_dir + continued_epoch_file_name

    @override
    @property
    def param_name(self) -> str:
        return 'epoch_count'

    @override
    def compute(self) -> int:
        # Does not get called when epoch_count is already set
        if self.prev_params is None:
            epoch = 0
            if getattr(self.params, 'filebased_continued_epoch_counting', False):
                try:
                    return int(read_all(self.path))
                except:  # noqa
                    pass
            return epoch

        return self.prev_value + 1

    @override
    def after_fit(self, history, x, y):
        overwrite_all(self.path, str(self.current_value))
        return super().after_fit(history, x, y)


class SetEpochIndexCallback(Callback):
    def __init__(self, params):
        self._params = params

    def on_epoch_end(self, batch, logs={}):
        self._params.epoch_count += 1


class ApplyInitialEpochAndNumEpochToFitArgsTrainerExtension(TrainerExtension):

    def before_fit(self, x, y):
        initial_epoch = getattr(self.params, 'epoch_count', 0)
        n_epochs = getattr(self.params, 'n_epochs')
        assert n_epochs is not None, "'n_epochs' is mandatory"

        self.params.fit_args.initial_epoch = initial_epoch
        self.params.fit_args.epochs = initial_epoch + n_epochs

        return x, y


class ContinuousStageCountExtensions(ComputeValueTrainerExtension):
    @override
    def initialize(self):
        super().initialize()
        overwrite_all(self.path, str(self.current_value + 1))  # + 1 converts index to count

    @override
    def compute(self) -> int:
        # Does not get called when epoch_count is already set
        if self.prev_params is None:
            if getattr(self.params, 'filebased_continued_stage_counting', False):
                try:
                    return int(read_all(self.path))
                except:  # noqa
                    pass
            return 0
        return self.prev_value + 1

    @override
    @property
    def param_name(self) -> str:
        return 'stage'

    @property
    def path(self) -> str:
        return self.params.log_dir + continued_stage_file_name
