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
from typeguard import check_argument_types, check_return_value  # noqa
# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
# from python.keyboard.hp import Params, MLModel
from trainer.ModelAdapter import ParameterizedCreateModelBase, Params as ParameterizeModelParams
from utilities import override


class Params(ParameterizeModelParams):
    max_timesteps: int
    swipe_feature_count: int


class ModelFactory(ParameterizedCreateModelBase):
    @property
    def params(self) -> Params:
        return self._params

    def __init__(self, params: Params):
        assert check_argument_types()
        # for param in required_params:
        #    assert param in params, f"Required param '{param}' missing"

        super().__init__(params)

    @override
    def _create_model(self) -> Model:
        # None here means variable over batches (but not within a batch)
        input = Input(shape=(self.params.max_timesteps, self.params.swipe_feature_count))
        masking = Masking(mask_value=myNaN)(input)

        d = Dense(50, kernel_initializer='random_uniform', activation='linear')(masking)
        lstm = LSTM(16, kernel_initializer='random_uniform')(d)
        middle = Dense(50, kernel_initializer='random_uniform', activation='relu')(lstm)
        output = Dense(1, kernel_initializer='random_uniform', activation='sigmoid')(middle)

        model = Model(inputs=[input], outputs=output)

        return model
