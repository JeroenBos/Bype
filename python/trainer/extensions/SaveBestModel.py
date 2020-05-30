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
    def __init__(self, 
                 filepath: str,         # passed to ModelCheckpoint.__init__
                 monitor='val_loss',    # passed to ModelCheckpoint.__init__
                 mode='auto',           # passed to ModelCheckpoint.__init__
                 best_over_all_stages=True,
                 ):
        assert isinstance(filepath, str) 
        assert filepath.endswith('.h5')

        self._best_over_all_stages = best_over_all_stages
        self._kw = {
            "filepath": filepath,
            "monitor": monitor,
            "mode": mode,
            "save_best_only": True,
        }

    def _find_previous_callback(self):

        return next((callback for callback in self.params.fit_args.callbacks 
                     if isinstance(callback, _SaveBestModelCallback) and callback.filepath == self.filepath), None)

    def initialize(self):
        callback = None
        if self._best_over_all_stages:
            callback = self._find_previous_callback()
            if callback is None and not self.is_first_stage:
                print("Failed to find previous save_best_model_callback")


        if callback is None:
            callback = _SaveBestModelCallback(**self._kw)

        self.params.fit_args.callbacks.append(callback)


class _SaveBestModelCallback(ModelCheckpoint):

    @override
    def _save_model(self, epoch, logs):
        super(self.__class__, self)._save_model(epoch, logs)
