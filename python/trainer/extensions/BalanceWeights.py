from typing import Any, Optional, Tuple
import numpy as np
from sklearn.utils import class_weight

from trainer.ModelAdapter import FitArgs
from trainer.trainer import TrainerExtension
from trainer.types import X, Y

class BalanceWeightsTrainerExtension(TrainerExtension):
    """
    Gives weights to the training data such that they sum up to equal weight for each y. 
    E.g. in binary classification where 10 training elements map to true and 20 to false, then the first set count doubly.
    """

    def before_fit(self, x: X, y: Y) -> Tuple[X, Y]:
        weights = class_weight.compute_class_weight("balanced", np.unique(y), y)
        self.params.fit_args.class_weight = weights
        return x, y
