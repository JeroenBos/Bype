import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard  # noqa
from typing import Callable, Union

from trainer.trainer import TrainerExtension, Trainer
from utilities import override, virtual
from trainer.ModelAdapter import ParamsBase


class TensorBoardExtension(TrainerExtension):
    def initialize(self):
        self.params.fit_args.callbacks.append(
            TensorBoard(log_dir=self.params.run_log_dir, 
                        histogram_freq=1,
                        profile_batch=100000000,  # prevents error "Must enable trace before export."
                        )
        )



