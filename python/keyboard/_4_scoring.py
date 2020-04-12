import numpy as np
from python.keyboard._0_types import SwipeEmbeddingDataFrame, Input, ProcessedInput
from python.keyboard._3_model import KeyboardEstimator
from typing import List
from collections import namedtuple
from itertools import count, takewhile
from tensorflow.keras.losses import Loss  # noqa
from python.keyboard._2_transform import Preprocessor


Convolution = namedtuple('Convolution', 'swipe correct')


class Scorer():
    """
    It returns a floating point number that quantifies the estimator prediction quality on X, with reference to y.
    Again, by convention higher numbers are better, so if your scorer returns loss, that value should be negated.
    """

    def __init__(self, trainings_data: SwipeEmbeddingDataFrame):
        assert SwipeEmbeddingDataFrame.is_instance(trainings_data), \
            f"Arg error: expected SwipeEmbeddingDataFrame; got '{str(type(trainings_data))}'"
        self.trainings_data = trainings_data

    def __call__(self, estimator: KeyboardEstimator, X: ProcessedInput, y: None) -> float:
        """
        :param X: Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
                  So this X should be (timesteps, features)?
        """

        word, correctSwipe = estimator.preprocessor.decode(X)
        swipes = (t.swipe for t in self.trainings_data)
        intermediate = (Convolution(self._predict(word, swipe), swipe == correctSwipe) for swipe in swipes)
        sortedIntermediate = sorted(intermediate, key=lambda t: t.convolution)
        result = count(takewhile(lambda t: not t.correct, sortedIntermediate))
        return result

    def _predict(self, word, swipe) -> float:
        return self.estimator.predict(self.estimator.preprocessor.encode(swipe, word))



class MyLoss(Loss):
    def __init__(self,
                 trainings_data: SwipeEmbeddingDataFrame,
                 name='myError'):
        super().__init__()
        self.scorer = Scorer(trainings_data)

    def __call__(self, preprocessor: Preprocessor) -> float:
        assert isinstance(preprocessor, Preprocessor)

        def score(**kwargs):
            return self.scorer.__call__(preprocessor, None)
        return score
