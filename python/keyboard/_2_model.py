from python.keyboard.hp import MyBaseEstimator, Model
from typing import List, Union, Optional  # noqa
import tensorflow as tf  # noqa
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate  # noqa
from python.keyboard._2a_word_input_model import CappedWordStrategy, WordStrategy

# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
# from python.keyboard.hp import Params, MLModel


class KeyboardEstimator(MyBaseEstimator):
    def __init__(self, num_epochs=5,
                 activation='relu',
                 swipe_feature_count=4,
                 word_input_strategy: WordStrategy = CappedWordStrategy(5)):
        super().__init__()
        self.num_epochs = num_epochs
        self.activation = activation
        self.swipe_feature_count = swipe_feature_count
        self.word_input_strategy = word_input_strategy

    def _create_model(self) -> Model:
        # None here means variable over batches (but not within a batch)
        swipe_input = Input(shape=(self.swipe_feature_count, None))
        if isinstance(self.word_input_strategy, CappedWordStrategy):
            word_input = Input(shape=(self.word_input_strategy.n, ))

        swipe_dense = Dense(10, )(swipe_input)
        word_dense = Dense(10, )(word_input)
        merged = concatenate([swipe_dense, word_dense])

        model = Model(inputs=[swipe_input, word_input], outputs=merged)
        return model

    def _compile(self, model: Optional[Model] = None) -> None:
        model = model if model else self.current_model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
