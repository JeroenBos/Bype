import math
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import Callback  # noqa
from sklearn.metrics import roc_auc_score
from keyboard._0_types import SwipeEmbeddingDataFrame, Input, ProcessedInput, SwipeDataFrame, SwipeConvolutionDataFrame
from typing import List, Callable
from collections import namedtuple
from itertools import count, takewhile
from tensorflow.keras.losses import Loss  # noqa
from keyboard._2_transform import Preprocessor
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution  # noqa
from tensorflow.python.framework import ops  # noqa
from tensorflow.python.ops import math_ops  # noqa
from tensorflow.python.keras import backend as K  # noqa



class Metrics(Callback):
    def __init__(self, preprocessed_convolved_validation_data, decode: Callable):
        super().__init__()
        self.test_data = preprocessed_convolved_validation_data
        self.decode = decode
        assert self._L ** 2 == len(self.test_data), "How come the convolved data isn't square?"

    @property
    def _L(self):
        return (int)(math.sqrt(len(self.test_data)))

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        L = self._L
        # assuming correct ones are on diagonal
        y = self.model.predict(self.test_data)
        y_predict = np.asarray(y)
        # place indicates howmaniest place the word would be suggested
        place = np.zeros(len(y_predict) // L, int)
        for i in range(len(y_predict)):
            correct_index = (i % L) * (L + 1)
            correct_word = self.decode(self.test_data[correct_index])
            input_word = self.decode(self.test_data[i])
            assert correct_word != input_word or correct_index == i
            if y_predict[i] >= y_predict[correct_index]:
                place[i // L] += 1
        print('\n' + str(place))

        # # create_swipe_embedding_df
        # word, correctSwipe = X
        # swipes = self.trainings_data.swipes
        # # for swipe in swipes:
        # #     assert SwipeDataFrame.is_instance(swipe)
        # intermediate = [Convolution(self._predict(estimator, swipe, word), swipe == correctSwipe) for swipe in swipes]
        # sortedIntermediate = sorted(intermediate, key=lambda t: t.convolution)
        # result = count(takewhile(lambda t: not t.correct, sortedIntermediate))
        # return result

        # X_val, y_val = self.validation_data[0], self.validation_data[1]
        # y_predict = np.asarray(self.model.predict(X_val))

        # y_val = np.argmax(y_val, axis=1)
        # y_predict = np.argmax(y_predict, axis=1)
        # print(' - metric: 0 %')
        self._data.append({
            'val_rocauc': 1,
        })
        return

    def get_data(self):
        return self._data


# disable_eager_execution()

Convolution = namedtuple('Convolution', 'swipe correct')



class Scorer():
    """
    It returns a floating point number that quantifies the estimator prediction quality on X, with reference to y.
    Again, by convention higher numbers are better, so if your scorer returns loss, that value should be negated.
    """

    def __init__(self, trainings_data: SwipeConvolutionDataFrame):
        assert SwipeEmbeddingDataFrame.is_instance(trainings_data), \
            f"Arg error: expected SwipeEmbeddingDataFrame; got '{str(type(trainings_data))}'"
        self.trainings_data = trainings_data

    def __call__(self, estimator: "KeyboardEstimator", X: Input, y: None) -> float:
        """
        :param X: Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
                  So this X should be (timesteps, features)?
        """

        prediction_matrix = estimator.predict(self.trainings_data)



    def _predict(self, estimator, swipe, word) -> float:
        return estimator.predict(estimator.preprocessor.encode(swipe, word))
