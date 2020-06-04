import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard  # noqa
from typing import Callable, Union

from trainer.trainer import TrainerExtension, Trainer
from utilities import override, virtual
from trainer.ModelAdapter import ParamsBase

class TensorBoardScalar(TrainerExtension):
    def __init__(self, 
                 prefix="",
                 log_dir_postfix='/metrics',
                 **named_scalars: Callable[[ParamsBase], Union[int, float]]):
        self._log_dir_postfix = log_dir_postfix
        self._named_scalars = {prefix + key: value for key, value in named_scalars.items()}


    @override
    def initialize(self):
        self.params.fit_args.callbacks.append(
            self._WriteTensorBoardScalarCallback(self.params, self._named_scalars, self._log_dir_postfix)
        )

    class _WriteTensorBoardScalarCallback(Callback):
        def __init__(self, params, scalar_functions, log_dir_postfix):
            super().__init__()
            self._log_dir_postfix = log_dir_postfix
            self._scalar_functions = scalar_functions
            self._params = params  # ref copy to params; can't be named params because Callback already declares that


        @property
        def _resource_writer_name(self) -> str:
            return self._params.log_dir + self._log_dir_postfix

        @property
        def _tf_summary_writer(self) -> tf.summary.SummaryWriter:
            return self._params.get_resource_writer(self._resource_writer_name)

        @override
        def on_epoch_end(self, batch, logs={}):
            with self._tf_summary_writer.as_default():
                for name, fn in self._scalar_functions.items():
                    value = fn(self._params)
                    tf.summary.scalar(name, data=value, step=batch)
                self._tf_summary_writer.flush()
