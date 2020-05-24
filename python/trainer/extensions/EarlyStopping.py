from trainer._trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import EarlyStopping  # noqa
from utilities import override


class EarlyStoppingTrainerExtension(TrainerExtension):
    def __init__(self, params):
        pass

class EarlyStoppingCallback(EarlyStopping):
    pass  # not implemented
