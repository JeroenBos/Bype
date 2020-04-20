import tensorflow
from typing import List


class LossHistory(tensorflow.keras.callbacks.Callback):
    def __init__(self, monitor='loss'):
        self.monitor = monitor
        self.losses: List[float] = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs=None):
        loss = logs.get(self.monitor)

        self.losses.append(loss)
