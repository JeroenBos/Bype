from dataclasses import dataclass
from trainer.trainer import TrainerExtension, Trainer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping  # noqa
from utilities import override


@dataclass
class Params:
    file_path: str
    monitor: str = 'loss'
    save_only_best: bool = False


class SaveBestModelTrainerExtension(TrainerExtension):
    def __init__(self, params: Params):
        assert isinstance(params.file_path, str) and params.file_path.endswith('.h5')

        self._log_dir = params.file_path
        self._monitor = params.monitor
        self._save_best_only = params.save_best_only

    def initialize(self, trainer: Trainer) -> None:
        trainer.callbacks.append(
            _SaveBestModelCallback(file_path=self._file_path,
                                   monitor=self._monitor, 
                                   save_best_only=self._save_best_only,
                                   )
        )


class _SaveBestModelCallback(ModelCheckpoint):
    def __init__(self, file_path):
        self._file_path = file_path

    @override
    def _save_model(self, epoch, logs):
        super(self.__class__, self)._save_model(epoch, logs)
        self.preprocessor.save(self.preprocessor_filepath)

    @property
    def preprocessor_filepath(self):
        return get_processor_path(self._filepath)

def get_processor_path(h5path: str):
    assert isinstance(h5path, str)
    assert h5path.endswith('.h5')
    return h5path[0:h5path.rindex('.')] + '.json'
