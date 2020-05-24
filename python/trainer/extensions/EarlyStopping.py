from trainer._trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import EarlyStopping  # noqa
from utilities import override
from typeguard import check_argument_types  # noqa


class EarlyStoppingTrainerExtension(TrainerExtension):
    def __init__(self, params):
        pass

class EarlyStoppingCallback(EarlyStopping):
    pass  # not implemented
