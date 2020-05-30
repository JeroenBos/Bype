from trainer.trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping  # noqa
from utilities import override


class TensorBoardExtension(TrainerExtension):
    def initialize(self):
        self.params.fit_args.callbacks.append(
            TensorBoard(log_dir=self.params.log_dir, 
                        histogram_freq=1,
                        profile_batch=100000000,  # prevents error "Must enable trace before export."
                        )
        )
