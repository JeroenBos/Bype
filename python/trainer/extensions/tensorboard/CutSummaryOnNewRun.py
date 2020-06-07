from unordereddataclass import mydataclass
from dataclasses import dataclass
import math
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard  # noqa
from typing import Callable, Union, Dict

from trainer.trainer import TrainerExtension, Trainer
from utilities import bind, override, print_in_red, virtual
from trainer.ModelAdapter import ParamsBase
from trainer.extensions.tensorboard.ResourceWriterPool import Params as ResourceWriterPoolParams
from dataclasses import field

@dataclass
class _Entry:
    name: str
    graph_name: str
    x: int

@mydataclass
class Params:
    also_cut_default_summaries: bool = True
    _summary_writer_last_epochs: Dict[str, _Entry] = field(default_factory=dict)

    @override
    def write_scalar(self: ResourceWriterPoolParams, name: str, x: int, y: Union[float, int], graph_name: str = ""):
        key = name + graph_name
        self._summary_writer_last_epochs[key] = _Entry(name=name, x=x, graph_name=graph_name)
        super(Params, self).write_scalar(name, x, y, graph_name)



class CutSummaryOnNewRun(TrainerExtension):
    """
    When you have multiple runs in tensorboard, some ugly lines from the end of the previous run to the next exist. 
    This extension removes that line.
    """
    @override
    def initialize(self):
        assert isinstance(self.params, ResourceWriterPoolParams), "This extension is pointless without the ResourceWriterPool.Params"
        assert isinstance(self.params, Params), "The params must derive from CutSummaryOnNewRunParams"
        super().initialize()

    @property
    def _default_logs(self):
        # TODO: fetch from somewhere instead
        return [
            _Entry(graph_name="epoch_loss", name="train", x=-1),
            _Entry(graph_name="epoch_accuracy", name="train", x=-1)
        ]

    def _get_entries(self):
        yield from self.params._summary_writer_last_epochs.values()
        if self.params.also_cut_default_summaries:
            yield from self._default_logs

    @override
    def cleanUp(self):
        for entry in self._get_entries():
            self.params._write_scalar(y=math.nan, **entry.__dict__)
        return super().cleanUp()
