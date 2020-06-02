import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard  # noqa
from typing import Callable, Union

from trainer.trainer import TrainerExtension, Trainer
from utilities import override, virtual
from trainer.ModelAdapter import ParamsBase


class TensorBoardExtension(TrainerExtension):
    def initialize(self):
        self.params.fit_args.callbacks.append(
            TensorBoard(log_dir=self.params.log_dir, 
                        histogram_freq=1,
                        profile_batch=100000000,  # prevents error "Must enable trace before export."
                        )
        )




class TensorBoardScalar(TrainerExtension):
    def __init__(self, 
                 prefix="",
                 log_dir_postfix='/metrics',
                 **named_scalars: Callable[[ParamsBase], Union[int, float]]):
        self._log_dir_postfix = log_dir_postfix
        self._named_scalars = {prefix + key: value for key, value in named_scalars.items()}

    @virtual
    @property
    def log_dir(self) -> str:
        return self.params.log_dir + self._log_dir_postfix

    @override
    def initialize(self):
        writer = tf.summary.create_file_writer(self.log_dir)
        self.params.fit_args.callbacks.append(
            self._WriteTensorBoardScalarCallback(self.params, writer, self._named_scalars)
        )

    class _WriteTensorBoardScalarCallback(Callback):
        def __init__(self, params, writer, scalar_functions):
            super().__init__()
            self._writer = writer
            self._scalar_functions = scalar_functions
            self._params = params  # ref copy to params

        @override
        def on_epoch_end(self, batch, logs={}):
            with self._writer.as_default():
                for name, fn in self._scalar_functions.items():
                    value = fn(self._params)
                    tf.summary.scalar(name, data=value, step=batch)
                self._writer.flush()
