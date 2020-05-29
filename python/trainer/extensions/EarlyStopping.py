from trainer.trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import EarlyStopping  # noqa
from utilities import override


class EarlyStoppingTrainerExtension(TrainerExtension):
    pass

class EarlyStoppingCallback(EarlyStopping):
    pass  # not implemented
