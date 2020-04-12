import numpy as np
from python.keyboard._0_types import SwipeEmbeddingDataFrame, Input, ProcessedInput, SwipeDataFrame
from python.keyboard._1a_generate import create_swipe_embedding_df
from python.keyboard._3_model import KeyboardEstimator
from typing import List
from collections import namedtuple
from itertools import count, takewhile
from tensorflow.keras.losses import Loss  # noqa
from python.keyboard._2_transform import Preprocessor
import pandas as pd

Convolution = namedtuple('Convolution', 'swipe correct')


class SwipeConvolutionDataFrame(SwipeEmbeddingDataFrame):
    correct: pd.Series  # series of booleans

class Scorer():
    """
    It returns a floating point number that quantifies the estimator prediction quality on X, with reference to y.
    Again, by convention higher numbers are better, so if your scorer returns loss, that value should be negated.
    """

    def __init__(self, trainings_data: SwipeEmbeddingDataFrame):
        assert SwipeEmbeddingDataFrame.is_instance(trainings_data), \
            f"Arg error: expected SwipeEmbeddingDataFrame; got '{str(type(trainings_data))}'"
        self.trainings_data = trainings_data
        self.convolved_data = Scorer.convolve_data(trainings_data)

    @staticmethod
    def convolve_data(trainings_data: SwipeEmbeddingDataFrame) -> SwipeConvolutionDataFrame:
        # just creates a square matrix of all combinations
        L = len(trainings_data.words)
        words = [trainings_data.words[i % L] for i in range(L * L)]
        return create_swipe_embedding_df(words, lambda word, i: trainings_data.swipes[i // L])

    def __call__(self, estimator: KeyboardEstimator, X: Input, y: None) -> float:
        """
        :param X: Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
                  So this X should be (timesteps, features)?
        """

        prediction_matrix = estimator.predict(self.convolved_data)

        create_swipe_embedding_df
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

            def __call__(self, y_true, y_pred, **kwargs):
                assert _get_shape(y_pred) == [None, estimator.preprocessor.swipe_timesteps_count, 1]  # is the output count
                self.score_called_count += 1
                if self.score_called_count == 1:
                    return 0
                current_batch = self._get_batch_X(y_true) 
                losses = scorer.__call__(estimator, current_batch, y_pred)
        return Score(self._get_batch_X)
