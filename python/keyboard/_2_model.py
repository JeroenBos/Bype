from python.keyboard.hp import MyBaseEstimator, Model
from typing import List, Union  # noqa
import tensorflow as tf


# from python.keyboard.hp import Params, MLModel
class KeyboardEstimator(MyBaseEstimator):
    def __init__(self, num_epochs=5, activation='relu'):
        super().__init__()
        self.num_epochs = num_epochs
        self.activation = activation

    def _create_model(self) -> Model:
        return tf.keras.Sequential([
                   tf.keras.layers.Dense(14, activation=self.activation),
                   tf.keras.layers.Dense(1, activation='sigmoid')
               ])
