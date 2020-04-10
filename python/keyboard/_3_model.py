from python.keyboard.hp import MyBaseEstimator, Models
from typing import List, Union, Optional
import tensorflow as tf
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate  # noqa
from python.keyboard._3a_word_input_model import CappedWordStrategy, WordStrategy

# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
# from python.keyboard.hp import Params, MLModel


class KeyboardEstimator(MyBaseEstimator):
    def __init__(self, num_epochs=5,
                 activation='relu',
                 swipe_feature_count=4,
                 swipe_timesteps_count: int = 100,
                 word_input_strategy: WordStrategy = CappedWordStrategy(5)):
        super().__init__()
        self.num_epochs = num_epochs
        self.activation = activation
        self.swipe_feature_count = swipe_feature_count
        self.word_input_strategy = word_input_strategy
        self.swipe_timesteps_count = swipe_timesteps_count

    def _create_model(self) -> Models:
        # None here means variable over batches (but not within a batch)
        swipe_input = Input(shape=(self.swipe_feature_count, self.swipe_timesteps_count))
        if isinstance(self.word_input_strategy, CappedWordStrategy):
            # this means the word input is appended to every timestep in the swipe data
            word_input = Input(shape=(self.word_input_strategy.n, self.swipe_timesteps_count))

        merged = concatenate([swipe_input, word_input], axis=1)

        model = Model(inputs=[swipe_input, word_input], outputs=merged)
        return model

    def _compile(self, model: Optional[Models] = None) -> None:
        model = model if model else self.current_model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
