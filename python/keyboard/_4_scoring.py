import numpy as np  # noqa
from python.keyboard._2_transform import data, Input, xType, encode, decode  # noqa
from python.keyboard._3_model import KeyboardEstimator
from typing import List
from collections import namedtuple
from itertools import count, takewhile


Convolution = namedtuple('Convolution', 'swipe correct')


class Scorer():
    """
    It returns a floating point number that quantifies the estimator prediction quality on X, with reference to y.
    Again, by convention higher numbers are better, so if your scorer returns loss, that value should be negated.
    """

    def __init__(self, trainings_data: List[Input]):
        assert isinstance(trainings_data, List), "Argument of wrong type. Expected a list"
        for element in trainings_data:
            assert isinstance(element, Input), "Argument of wrong type. Expected a list of Inputs"
        self.trainings_data = trainings_data

    def __call__(self, estimator: KeyboardEstimator, X: xType, y: None) -> float:
        """
        :param X: Input to an LSTM layer always has the (batch_size, timesteps, features) shape.
                  So this X should be (timesteps, features)?
        """

        # int wrongCounts = data.labeledSwipes
        #               .Select(_ => _.Swipe)
        #               .Select(swipe => (Convolution: model(X.word, swipe), Correct: swipe == X.Swipe))
        #               .OrderBy(_ => _.Convolution)
        #               .TakeWhile(_ => !_.Correct)
        #               .Count();
        print(type(X))
        print(type(y))
        word, correctSwipe = decode(X)
        swipes = (t.swipe for t in self.trainings_data)
        intermediate = (Convolution(self._predict(word, swipe), swipe == correctSwipe) for swipe in swipes)
        sortedIntermediate = sorted(intermediate, key=lambda t: t.convolution)
        result = count(takewhile(lambda t: not t.correct, sortedIntermediate))
        return result

    def _predict(self, word, swipe) -> float:
        return self.estimator.predict(encode(word, swipe))


score_function = Scorer(data)


# class MyError(losses.LossFunctionWrapper):
#     def __init__(self, model: Model, reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE, name='myError'):
#         self.model = model
#         super(MyError, self).__init__(lambda y_true, y(pred: self.get_error(y_true, y_pred),
#                                       name=name,
#                                       reduction=reduction)

#     def get_error(self, y_true, y_pred):
#         return 0
