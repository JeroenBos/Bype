from MyBaseEstimator import MyBaseEstimator
from typing import List, Union, Optional, Callable
import tensorflow as tf
from tensorflow.keras import Model  # noqa
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate  # noqa
from keyboard._0_types import SwipeEmbeddingDataFrame, SwipeDataFrame, Input as EmbeddingInput
from keyboard._2_transform import Preprocessor
from keyboard._3a_word_input_model import CappedWordStrategy, WordStrategy
from generic import generic
from tensorflow.keras.losses import Loss  # noqa
from tensorflow.python.keras import layers, models  # noqa
from DataSource import DataSource
# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
# from python.keyboard.hp import Params, MLModel


# The whole idea behind the metaclass is that I create a new instance of this keyboardestimator class,
# each of which is associated with a preprocessor. sklearn clones the estimator, but doesn't tag 
# along the preprocessor because it's not an hp (it would do a deepclone anyway, which I guess itn't that bad)
# but the metaclass does tag along the preprocessor then (different one per KeyboardEstimator type, like I said)
class KeyboardEstimator(MyBaseEstimator, metaclass=generic('preprocessor')):
    preprocessor: Preprocessor

    @classmethod
    def create_initialized(cls, **keyboard__init__kwargs):
        """ Creates a keyboard estimator and initializes the parameters from the preprocessor"""
        result = cls(**keyboard__init__kwargs)
        keys = list(result.get_params().keys())
        ignored_params = [key for key in keys 
                          if key not in cls.preprocessor.__dict__ and key not in keyboard__init__kwargs.keys()]
        if len(ignored_params) != 0:
            print("Attributes not directly specified and not copied from preprocessor because it doesn't have them: "
                  + str(ignored_params))

        duplicate_kwargs = [key for key in keyboard__init__kwargs if key in cls.preprocessor.__dict__]
        if len(duplicate_kwargs) != 0:
            print("The following keyboard attributes where specified to KeyboardEstimator.__init__, "
                  + "but should probably be specified on the preprocessor, because it has them too: "
                  + str(duplicate_kwargs))
        params = {key: cls.preprocessor.__dict__[key] for key in keys if key in cls.preprocessor.__dict__}
        result.set_params(**params)
        return result

    def __init__(self, 
                 num_epochs=5,
                 activation='relu',
                 word_input_strategy: WordStrategy = CappedWordStrategy(5),
                 loss_ctor: Union[str, Callable[["KeyboardEstimator"], Loss]] = 'binary_crossentropy'):
        super(self.__class__, self).__init__()
        self.num_epochs = num_epochs
        self.activation = activation
        self.word_input_strategy = word_input_strategy
        self.loss_ctor = loss_ctor

    def _create_model(self) -> Model:
        # None here means variable over batches (but not within a batch)
        input = Input(shape=(self.swipe_timesteps_count, self.swipe_feature_count))

        a = LSTM(64, kernel_initializer='random_uniform')(input)
        middle = Dense(20, kernel_initializer='random_uniform')(a)
        output = Dense(1, kernel_initializer='random_uniform', activation='sigmoid')(middle)

        model = Model(inputs=[input], outputs=output)
        return model

    # called by super
    def _preprocess(self, X):
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            return self.preprocessor.preprocess(X)
        return super(self.__class__, self)._preprocess(X)

    def _compile(self, model: Optional[Model] = None) -> None:
        model = model if model else self.current_model
        loss = self.loss_ctor if isinstance(self.loss_ctor, str) else self.loss_ctor(self)
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])

    def set_params(self, **params):
        self.preprocessor.set_params(**params)
        return super(self.__class__, self).set_params(**params)

    @property
    def swipe_feature_count(self):
        return self.preprocessor.swipe_feature_count

    @property
    def swipe_timesteps_count(self):
        return self.preprocessor.swipe_timesteps_count

    def fit(self, X, y=None) -> None:
        if y is None and isinstance(X, DataSource):
            X, y = X.get_train(), X.get_target()
        assert y.name == 'correct'
        assert len(X) == len(y)
        assert all(isinstance(target, float) for target in y)
        super(self.__class__, self).fit(X, y)
