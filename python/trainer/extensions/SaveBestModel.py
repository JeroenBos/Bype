from dataclasses import dataclass
from typing import Callable, Union
from trainer.trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping  # noqa
from utilities import override
import os


@dataclass
class Params:
    file_path: str
    monitor: str = 'loss'
    save_only_best: bool = False


class SaveBestModelTrainerExtension(TrainerExtension):
    def __init__(self, 
                 filepath: Union[str, Callable[[Params], str]],   # passed to ModelCheckpoint.__init__
                 monitor='val_loss',                              # passed to ModelCheckpoint.__init__
                 mode='auto',                                     # passed to ModelCheckpoint.__init__
                 best_over_all_stages=True,
                 ):
        assert isinstance(filepath, (str, Callable)) 

        self._best_over_all_stages = best_over_all_stages
        self._kw = {
            "filepath_lambda": filepath if isinstance(filepath, Callable) else lambda p: filepath,
            "monitor": monitor,
            "mode": mode,
            "save_best_only": True,
        }

    def _find_previous_callback(self):
        # what is the point of this again? Shouldn't it be `self.prev_params` btw?
        return next((callback for callback in self.params.fit_args.callbacks 
                     if isinstance(callback, _SaveBestModelCallback) and callback.filepath == self._kw.get("filepath", "")), None)

    def initialize(self):
        self._invoke_filepath_lambda()
        callback = None
        if self._best_over_all_stages:
            callback = self._find_previous_callback()
            if callback is None and not self.is_first_stage:
                print("Failed to find previous save_best_model_callback")

        if callback is None:
            callback = _SaveBestModelCallback(**self._kw)

        self.params.fit_args.callbacks.append(callback)

    def _invoke_filepath_lambda(self):
        self._kw["filepath"] = self._kw["filepath_lambda"](self.params)
        assert self._kw["filepath"].endswith(".h5"), "Should end with .h5"
        del self._kw["filepath_lambda"]



class _SaveBestModelCallback(ModelCheckpoint):

    @override
    def _save_model(self, epoch, logs):
        if self.monitor in logs:  # prevents the warning that '<monitor>' can't be found
            super(self.__class__, self)._save_model(epoch, logs)
