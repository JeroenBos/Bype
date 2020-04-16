from tensorflow.keras.callbacks import ModelCheckpoint  # noqa
from generic import generic

class MyModelCheckpoint(ModelCheckpoint, metaclass=generic('preprocessor')):
    """
    A model checkpoint which also saves preprocessor attributes in a similarly named json file.
    """

    # override
    def _save_model(self, epoch, logs):
        super(self.__class__, self)._save_model(epoch, logs)
        self.preprocessor.save(self.preprocessor_filepath)

    @property
    def preprocessor_filepath(self):
        return self.filepath[0:self.filepath.rindex('.')] + '.json'
