from datetime import datetime
from os import path
from typing import Callable, Optional, Tuple, Union

from trainer.types import TrainerExtension, IModel
from trainer.ModelAdapter import ParamsBase


class TagWithTimestampTrainerExtension(TrainerExtension):
    def initialize(self):
        tag_format = getattr(self.params, 'tag', None)
        tag_format = tag_format if tag_format and not tag_format.isspace() else "%Y_%m_%d"

        setattr(self.params, 'tag', datetime.now().strftime(tag_format))

class LogDirPerDataTrainerExtension(TrainerExtension):
    """ Renames the current log_dir to `{log_dir}/{current_date}`. """


    def initialize(self):
        assert hasattr(self.params, "log_dir")
        super().initialize()

        current_log_dir = path.join(self.params.log_dir, "%Y_%m_%d")
        current_log_dir = datetime.now().strftime(current_log_dir)
        self.params.log_dir = current_log_dir


# this has to be a separate extension from LogDir because this uses the stage.
# however, the extension that sets the stage must know the log_dir (to find the file that might contain the current stage)
# but it doesn't need to know the run_log_dir (which depends on the stage), that's why it is set later (and thus separately)

class RunLogDirTrainerExtension(TrainerExtension):
    """ Renames the current run_log_dir to `{log_dir}/{run}`. """

    def __init__(self, get_run: Callable[[ParamsBase], Union[int, str]] = (lambda p: 'stage' + str(p.stage))):
        """ 
        :param get_run: A function that get a key describing which run is currently running.
        Tensorboard summaries will be appended there.
        """
        self._get_run = get_run
        super().__init__()

    @property
    def run(self):
        return str(self._get_run(self.params))


    def initialize(self):
        assert hasattr(self.params, "log_dir")
        super().initialize()

        setattr(self.params, "run_log_dir", path.join(self.params.log_dir, self.run))
