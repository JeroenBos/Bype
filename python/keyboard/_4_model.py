from MyBaseEstimator import MyBaseEstimator
from typing import List, Union, Optional, Callable, Type
import tensorflow as tf
from tensorflow.keras.callbacks import Callback  # noqa
from tensorflow.keras import Model  # noqa
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Masking  # noqa
from tensorflow.keras.optimizers import Adam  # noqa
from tensorflow.keras.callbacks import History  # noqa
from keyboard._0_types import myNaN, SwipeEmbeddingDataFrame, SwipeDataFrame, Input as EmbeddingInput
from keyboard._2_transform import Preprocessor
from keyboard._4a_word_input_model import CappedWordStrategy, WordStrategy
from keyboard._4b_initial_weights import WeightInitStrategy
from generic import generic
from tensorflow.keras.losses import Loss  # noqa
from tensorflow.python.keras import layers, models  # noqa
from DataSource import DataSource
from keyboard.MyModelCheckpoint import MyModelCheckpoint
# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
# from python.keyboard.hp import Params, MLModel


# The whole idea behind the metaclass is that I create a new instance of this keyboardestimator class,
# each of which is associated with a preprocessor. sklearn clones the estimator, but doesn't tag 
# along the preprocessor because it's not an hp (it would do a deepclone anyway, which I guess itn't that bad)
# but the metaclass does tag along the preprocessor then (different one per KeyboardEstimator type, like I said)
class KeyboardEstimator(MyBaseEstimator, metaclass=generic('preprocessor')):
    preprocessor: Preprocessor
    extra_callbacks: List[Callback] = []

    def with_callback(self, *callbacks: Callback) -> "KeyboardEstimator":
        for callback in callbacks:
            self.extra_callbacks.append(callback)
        return self

    def __new__(cls, *args, **kwargs):
        cls.TModelCheckpoint = MyModelCheckpoint[cls.preprocessor]
        return super(cls.__class__, cls).__new__(cls)

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
                 initial_epoch=0,
                 activation='relu',
                 weight_init_strategy: WeightInitStrategy = WeightInitStrategy.no_init,
                 word_input_strategy: WordStrategy = CappedWordStrategy(5),
                 loss_ctor: Union[str, Callable[["KeyboardEstimator"], Loss]] = 'binary_crossentropy'):
        super(self.__class__, self).__init__()
        self.num_epochs = num_epochs
        self.initial_epoch = initial_epoch
        self.activation = activation
        self.word_input_strategy = word_input_strategy
        self.loss_ctor = loss_ctor
        self.weight_init_strategy = weight_init_strategy

    def _create_model(self) -> Model:
        # None here means variable over batches (but not within a batch)
        input = Input(shape=(self.max_timesteps, self.swipe_feature_count))
        masking = Masking(mask_value=myNaN)(input)

        d = Dense(50, kernel_initializer='random_uniform', activation='linear')(masking)
        lstm = LSTM(16, kernel_initializer='random_uniform')(d)
        middle = Dense(50, kernel_initializer='random_uniform', activation='relu')(lstm)
        output = Dense(1, kernel_initializer='random_uniform', activation='sigmoid')(middle)

        model = Model(inputs=[input], outputs=output)

        self.params['weight_init_strategy'].init_weights(model)

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
                      optimizer=Adam(),
                      metrics=['accuracy'])

    def set_params(self, **params):
        self.preprocessor.set_params(**params)
        return super(self.__class__, self).set_params(**params)

    @property
    def swipe_feature_count(self):
        return self.preprocessor.swipe_feature_count

    @property
    def max_timesteps(self):
        return self.preprocessor.max_timesteps

    def fit(self, X, y=None) -> History:
        if y is None and isinstance(X, DataSource):
            X, y = X.get_train(), X.get_target()
        assert y.name == 'correct'
        assert len(X) == len(y)
        assert all(isinstance(target, float) for target in y)
        return super(self.__class__, self).fit(X, y, extra_callbacks=self.extra_callbacks)
