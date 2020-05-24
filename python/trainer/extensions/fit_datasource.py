from typing import Tuple

from trainer._trainer import TrainerExtension, Trainer, X, Y
from DataSource import DataSource


class AllowDataSources(TrainerExtension):
    """ Allows a datasource to be specified as `x` in `fit(..)` """

    def before_fit(self, x: X, y: Y) -> Tuple[X, Y]:
        if y is None and isinstance(x, DataSource):
            return x.get_train(), x.get_target()
        else:
            return x, y
