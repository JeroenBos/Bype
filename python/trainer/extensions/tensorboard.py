from trainer._trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping  # noqa
from utilities import override


class TensorBoardExtension(TrainerExtension):
    def __init__(self, params):
        params.fit_args.callbacks.append(
            TensorBoard(log_dir=params.log_dir, histogram_freq=1)
        )
