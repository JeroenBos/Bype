from python.keyboard.hp import MyBaseEstimator, Model
from typing import List, Union, Optional  # noqa
import tensorflow as tf


# Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
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

    def _compile(self, model: Optional[Model] = None) -> None:
        model = model if model else self.current_model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


def sliding_windows(inputs):
    target, query = inputs
    target_length = tf.shape(target)[1]  # variable-length sequence, shape is a TF tensor
    query_length = tf.int_shape(query)[1]
    num_windows = target_length - query_length + 1  # number of windows is also variable

    # slice the target into consecutive windows
    start_indices = tf.arange(num_windows)
    windows = tf.map_fn(lambda t: target[:, t:(t + query_length), :],
                        start_indices,
                        dtype=tf.floatx())

    # `windows` is a tensor of shape (num_windows, batch_size, query_length, ...)
    # so we need to change the batch axis back to axis 0
    windows = tf.permute_dimensions(windows, (1, 0, 2, 3))

    # repeat query for `num_windows` times so that it could be merged with `windows` later
    query = tf.expand_dims(query, 1)
    query = tf.tile(query, [1, num_windows, 1, 1])

    # just a hack to force the dimensions 2 to be known (required by Flatten layer)
    windows = tf.reshape(windows, shape=tf.shape(query))
    return [windows, query]
