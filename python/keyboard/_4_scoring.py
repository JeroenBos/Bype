import numpy as np
from python.keyboard._0_types import SwipeEmbeddingDataFrame, Input, ProcessedInput, SwipeDataFrame, SwipeConvolutionDataFrame
from python.keyboard._3_model import KeyboardEstimator
from typing import List
from collections import namedtuple
from itertools import count, takewhile
from tensorflow.keras.losses import Loss  # noqa
from python.keyboard._2_transform import Preprocessor
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution  # noqa
from tensorflow.python.framework import ops  # noqa
from tensorflow.python.ops import math_ops  # noqa
from tensorflow.python.keras import backend as K  # noqa


disable_eager_execution()

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

    def __call__(self, estimator: KeyboardEstimator, X: Input, y: None) -> float:
        """
        :param X: Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
                  So this X should be (timesteps, features)?
        """

        prediction_matrix = estimator.predict(self.trainings_data)

        # create_swipe_embedding_df
        word, correctSwipe = X
        swipes = self.trainings_data.swipes
        for swipe in swipes:
            assert SwipeDataFrame.is_instance(swipe)
        intermediate = [Convolution(self._predict(estimator, swipe, word), swipe == correctSwipe) for swipe in swipes]
        sortedIntermediate = sorted(intermediate, key=lambda t: t.convolution)
        result = count(takewhile(lambda t: not t.correct, sortedIntermediate))
        return result

    def _predict(self, estimator, swipe, word) -> float:
        return estimator.predict(estimator.preprocessor.encode(swipe, word))

def _get_shape(tensor) -> List:
    """ Gets the shape in a list form instead of tensorshape object"""
    return [dim.value for dim in tensor.shape.dims]



class MyLoss(Loss):
    def __init__(self,
                 trainings_data: SwipeEmbeddingDataFrame,
                 name='myError'):
        super().__init__()
        self.trainings_data = trainings_data

    def _get_batch_X(self, y_true):
        return self.trainings_data

    def __call__(self, estimator: KeyboardEstimator) -> float:
        assert isinstance(estimator.preprocessor, Preprocessor)
        scorer = Scorer(self.trainings_data)

        class Score:
            def __init__(self, _get_batch_X):
                self.score_called_count = 0
                self._get_batch_X = _get_batch_X        

            @tf.function
            def __call__(self, y_true, y_pred, **kwargs):
                y_pred = ops.convert_to_tensor_v2(y_pred)
                y_true = math_ops.cast(y_true, y_pred.dtype)
                return K.mean(math_ops.abs(y_pred - y_true), axis=-1)

                # return tf.keras.backend.zeros(

                assert _get_shape(y_pred) == [None, estimator.preprocessor.swipe_timesteps_count, 1]  # is the output count
                self.score_called_count += 1
                if self.score_called_count == 1:
                    return 0
                current_batch = self._get_batch_X(y_true) 
                losses = scorer.__call__(estimator, current_batch, y_pred)
        return Score(self._get_batch_X)
