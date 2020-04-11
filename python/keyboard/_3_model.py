from python.keyboard.hp import MyBaseEstimator, Models
from typing import List, Union, Optional
import tensorflow as tf
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate  # noqa
from python.keyboard._0_types import SwipeEmbeddingDataFrame, SwipeDataFrame, Input as EmbeddingInput
from python.keyboard._2_transform import encode
from python.keyboard._3a_word_input_model import CappedWordStrategy, WordStrategy

# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
# from python.keyboard.hp import Params, MLModel


class Preprocessor:
    def __init__(self, word_input_strategy: WordStrategy = CappedWordStrategy(5)):
        self.swipe_feature_count = 4
        self.swipe_timesteps_count = 100
        self.word_input_strategy = word_input_strategy

    def preprocess(self, X: EmbeddingInput):
        return self._preprocess()

    def _preprocess(self, X: EmbeddingInput):
        assert EmbeddingInput.is_instance(X), "Maybe sklearn gives an element to X instead of the entire set?"

        if isinstance(self.word_input_strategy, CappedWordStrategy):
            # this means the word input is appended to every timestep in the swipe data
            return encode(X.swipe, X.word)
        else:
            raise ValueError()


class KeyboardEstimator(MyBaseEstimator):
    @staticmethod
    def create(preprocessor: Preprocessor, **kwargs) -> "KeyboardEstimator":
        result = KeyboardEstimator(**preprocessor.__dict__, **kwargs)
        result.preprocessor = preprocessor
        return result
    
    def __init__(self, 
                 num_epochs=5,
                 activation='relu',
                 word_input_strategy: WordStrategy = CappedWordStrategy(5)):
        super().__init__()
        self.num_epochs = num_epochs
        self.activation = activation
        self.word_input_strategy = word_input_strategy

    def _create_model(self) -> Models:
        # None here means variable over batches (but not within a batch)
        swipe_input = Input(shape=(self.swipe_feature_count, self.swipe_timesteps_count))
        if isinstance(self.word_input_strategy, CappedWordStrategy):
            # this means the word input is appended to every timestep in the swipe data
            word_input = Input(shape=(self.word_input_strategy.n, self.swipe_timesteps_count))
        else:
            raise ValueError()

        merged = concatenate([swipe_input, word_input], axis=1)

        model = Model(inputs=[swipe_input, word_input], outputs=merged)
        return model

    # called by super
    def _preprocess(self, X):
        if hasattr(self, 'preprocessor'):
            return self.preprocessor.preprocess(X)
        return super()._preprocess(X)

    def _compile(self, model: Optional[Models] = None) -> None:
        model = model if model else self.current_model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    @property
    def swipe_feature_count(self):
        return self.preprocessor.swipe_feature_count

    @property
    def swipe_timesteps_count(self):
        return self.preprocessor.swipe_timesteps_count
