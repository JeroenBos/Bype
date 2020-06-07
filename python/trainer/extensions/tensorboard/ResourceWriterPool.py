import math
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard  # noqa
from typing import Callable, Union

from trainer.trainer import TrainerExtension, Trainer
from utilities import override, print_in_red, virtual
from trainer.ModelAdapter import ParamsBase


class Params:
    def get_resource_writer(self, dir: str, index=0) -> tf.summary.SummaryWriter:
        """
        Gets a summary writer for the specified name. Creates one if it doesn't exist.
        :param dir: Specify the same name here when you want to append to the same graph.

        So tensorflow works VERY unintuitively. To add 2 lines to one graph you have to do this:

        > with get_resource_writer("Line1").as_default():
        >    tf.summary.scalar("Graph", data=data, epoch=epoch)
        >
        > with get_resource_writer("Line2").as_default():
        >    tf.summary.scalar("Graph", data=data, epoch=epoch)
        """
        if not hasattr(self, "_tf_resource_writers"):
            self._tf_resource_writers = {}
        key = dir + str(index)
        if key not in self._tf_resource_writers:
            print(dir)
            self._tf_resource_writers[key] = tf.summary.create_file_writer(dir)
        return self._tf_resource_writers[key]

    def write_scalar(self, name: str, x: int, y: Union[float, int], graph_name: str = "", quiet=False):
        """
        Because writing to a tensorflow summary is so complicated, I decided to make function out of it

        :param name: The name of the collection of data points to which you want to add this point. 
                     In a multiline graph, think of this as the name of the line.
        :param x: The x-coodinate of the point to draw, usually the epoch/batch argument.
        :param y: The y-coodinate of the point to draw, usually the value of interest.
        :param graph_name: The name of the graph to draw in (Optional). 
                           If no value is provided, the data collection will be in a separate graph.
        """
        assert isinstance(name, str)
        assert isinstance(x, int) or math.isnan(x)
        assert isinstance(y, (float, int))
        if math.isnan(y):
            if not quiet:
                print_in_red("Summary value is NaN")
            return False

        assert isinstance(graph_name, str)

        self._write_scalar(name, x, y, graph_name)

    def _write_scalar(self, name: str, x: int, y: Union[float, int], graph_name: str = ""):
        # don't get me started on `name` vs `graph_name`
        with self.get_resource_writer(os.path.join(self.run_log_dir, name)).as_default():
            tf.summary.scalar(graph_name, y, x)





class ResourceWriterPool(TrainerExtension):
    """
    This extension enables the summary writer extensions. 
    Those extensions cannot be responsible for the construction of the `tf.summary.SummaryWriter`s,
    because on name collisions they don't work (multi-threading is not supported).
    So this extension provides a cache of summary writers. 
    """

    def initialize(self):
        assert isinstance(self.params, Params)
        self.params.fit_args.callbacks.append(self._PeriodicFlushCallback(self.params))

    def cleanUp(self):
        for resource_writer in getattr(self.params, "_tf_resource_writers", {}).values():
            resource_writer.close()

    class _PeriodicFlushCallback(Callback):
        def __init__(self, params, every_n_epochs: int = 1):
            assert isinstance(every_n_epochs, int) and every_n_epochs > 0
            super().__init__()

            self._every_n_epochs = every_n_epochs
            self._params = params

        @override
        def on_epoch_end(self, epoch, logs={}):
            super().on_epoch_end()
            if (epoch % self._every_n_epochs) == 0:
                self._flush()

        def _flush(self):
            for resource_writer in getattr(self, "_tf_resource_writers", {}).values():
                resource_writer.flush()
