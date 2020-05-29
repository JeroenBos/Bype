from datetime import datetime
from typing import Tuple, Optional

from trainer.types import TrainerExtension, IModel


class TagWithTimestampTrainerExtension(TrainerExtension):
    def initialize(self):
        tag_format = getattr(self.params, 'tag', None)
        tag_format = tag_format if tag_format and not tag_format.isspace() else "%Y_%m_%d"

        setattr(self.params, 'tag', datetime.now().strftime(tag_format))

class LogDirPerDataTrainerExtension(TrainerExtension):
    """ Renames the current log_dir to `{log_dir}/{current_date}`. """
    def initialize(self):
        current_log_dir = getattr(self.params, 'log_dir', '') + "%Y_%m_%d/"
        setattr(self.params, 'log_dir', datetime.now().strftime(current_log_dir))
