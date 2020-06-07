from dataclasses import dataclass
from typing import Callable
from utilities import override
from tensorflow.keras.callbacks import Callback  # noqa
from trainer.trainer import TrainerExtension, Trainer
from keyboard._2_transform import Preprocessor

@dataclass
class Params:
    run_log_dir: str
    best_model_path: str
    preprocessor: Preprocessor

class SavePreprocessorTrainerExtension(TrainerExtension):
    def initialize(self):
        assert isinstance(self.params.run_log_dir, str)
        assert isinstance(self.params.best_model_path, str)
        assert isinstance(self.params.preprocessor, Preprocessor)

        preprocessor_path = get_processor_path(self.params.run_log_dir + self.params.best_model_path)
        self.params.fit_args.callbacks.append(
            _SavePreprocessorCallback(
                file_path=preprocessor_path,
                preprocessor=self.params.preprocessor,
            )
        )


def get_processor_path(h5path: str):
    assert isinstance(h5path, str)
    if '.' not in h5path:
        return h5path + '.json'
    assert h5path.endswith('.h5')
    return h5path[0:h5path.rindex('.')] + '.json'



class _SavePreprocessorCallback(Callback):
    def __init__(self, path, preprocessor: Preprocessor):
        self._path = path
        self._preprocessor = preprocessor

    @override
    def on_train_batch_begin(self, batch, logs=None):
        self.preprocessor.save(self.preprocessor_filepath)

    @property
    def preprocessor_filepath(self):
        return get_processor_path(self.filepath)
