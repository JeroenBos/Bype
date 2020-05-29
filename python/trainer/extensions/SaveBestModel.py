from dataclasses import dataclass
from typing import Callable
from trainer.trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping  # noqa
from utilities import override


@dataclass
class Params:
    file_path: str
    monitor: str = 'loss'
    save_only_best: bool = False


class SaveBestModelTrainerExtension(TrainerExtension):
    def initialize(self):
        assert isinstance(self.params.best_model_path, str) 
        assert self.params.best_model_path.endswith('.h5')

        self.params.fit_args.callbacks.append(
            _SaveBestModelCallback(
                file_path=self.params.best_model_path,
                #  monitor=self.params.monitor, 
                #  save_best_only=self.params.save_best_only,
            )
        )



class _SaveBestModelCallback(ModelCheckpoint):
    def __init__(self, file_path: str, monitor: str = "loss"):
        super().__init__(filepath=file_path, 
                         monitor=monitor,
                         save_best_only=True,
                         )
        self._file_path = file_path

    @override
    def _save_model(self, epoch, logs):
        super(self.__class__, self)._save_model(epoch, logs)
