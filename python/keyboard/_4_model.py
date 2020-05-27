from unordereddataclass import mydataclass
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
from generic import generic
from tensorflow.keras.losses import Loss  # noqa
from tensorflow.python.keras import layers, models  # noqa
from DataSource import DataSource
# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
# from python.keyboard.hp import Params, MLModel
from trainer.ModelAdapter import Params as ParameterizeModelParams
from utilities import override
from trainer.types import IModel
from trainer.trainer import TrainerExtension


@mydataclass
class Params:
    max_timesteps: int
    swipe_feature_count: int


class ModelFactory(TrainerExtension):
    def __init__(self, params):
        self.params = params

    @override
    def create_model(self, model: Optional[IModel]) -> IModel:
        assert model is None
        # None here means variable over batches (but not within a batch)
        input = Input(shape=(self.params.max_timesteps, self.params.swipe_feature_count))
        masking = Masking(mask_value=myNaN)(input)

        d = Dense(50, kernel_initializer='random_uniform', activation='linear')(masking)
        lstm = LSTM(16, kernel_initializer='random_uniform')(d)
        middle = Dense(50, kernel_initializer='random_uniform', activation='relu')(lstm)
        output = Dense(1, kernel_initializer='random_uniform', activation='sigmoid')(middle)

        model = Model(inputs=[input], outputs=output)

        return model
