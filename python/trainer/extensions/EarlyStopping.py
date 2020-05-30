from unordereddataclass import mydataclass
from trainer.trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import EarlyStopping  # noqa
from typing import Callable
from utilities import override

@mydataclass
class Params:
    pass
    # abort: bool = False  # super annoying that I have to repeat the default here (TODO) Actually, it's probably duplicating? :S So let's just uncomment


class EarlyStoppingTrainerExtension(TrainerExtension):
    def __init__(self, 
                 patience=0,                    # passed to EarlyStopping.__init__
                 monitor='val_loss',            # passed to EarlyStopping.__init__
                 min_delta=0,                   # passed to EarlyStopping.__init__
                 mode='auto',                   # passed to EarlyStopping.__init__ 
                 baseline=None,                 # passed to EarlyStopping.__init__
                 cancel_all_stages=False):
        self.cancel_all_stages = cancel_all_stages
        self._kw = {
            "patience": patience,
            "monitor": monitor,
            "min_delta": min_delta,
            "mode": mode,
            "baseline": baseline,
        }



    def initialize(self):
        callback = _EarlyStoppingCallback(
            on_early_stopped_callback=self._on_early_stopped,
            **self._kw
        )
        self.params.fit_args.callbacks.append(callback)

    def _on_early_stopped(self, stopped_epoch: int) -> None:
        if self.cancel_all_stages:
            self.params.abort = True

class _EarlyStoppingCallback(EarlyStopping):
    def __init__(self, on_early_stopped_callback: Callable[[int], None], **kw):
        super().__init__(**kw)
        self._on_early_stopped = on_early_stopped_callback

    @override
    def on_train_end(self, logs={}):
        if self.stopped_epoch != 0:
            self._on_early_stopped(self.stopped_epoch)
