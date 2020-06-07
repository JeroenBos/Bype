import math
from itertools import islice
import pandas as pd
import numpy as np
from numpy.ma.core import MaskedArray
from tensorflow.keras.callbacks import Callback  # noqa
from keyboard._0_types import SwipeEmbeddingDataFrame, Input, ProcessedInput, SwipeDataFrame, SwipeConvolutionDataFrame
from typing import List, Callable, Tuple, Union
from collections import namedtuple
from itertools import count, takewhile
from tensorflow.keras.losses import Loss  # noqa
from keyboard._2_transform import Preprocessor
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution  # noqa
from tensorflow.python.framework import ops  # noqa
from tensorflow.python.ops import math_ops  # noqa
from tensorflow.python.keras import backend as K  # noqa
from utilities import override, append_masked


class ValidationData:
    def __init__(self, unconvolved_data: SwipeEmbeddingDataFrame, preprocessor: Preprocessor):
        assert isinstance(unconvolved_data, SwipeEmbeddingDataFrame)
        self._inverse_indices = []
        self._unencoded_convolved_data_words = []
        self._encoded_convolved_data = None  # set in `self.add`

        self.add(unconvolved_data, preprocessor)

        assert len(self._inverse_indices) == len(self._unencoded_convolved_data_words)
        assert self._encoded_convolved_data.shape[0] == len(self._unencoded_convolved_data_words)

    def add(self, unconvolved_data: SwipeEmbeddingDataFrame, preprocessor: Preprocessor):
        _inverse_indices = []
        _unencoded_convolved_data = unconvolved_data.convolve(preprocessor.convolution_fraction,
                                                              inverse_indices_out=_inverse_indices)

        offset = (len(self._unencoded_convolved_data_words), len(self._unencoded_convolved_data_words), len(self._inverse_indices))
        self._inverse_indices.extend((i[0] + offset[0], i[1] + offset[1], i[2] + offset[2]) for i in _inverse_indices)

        _unencoded_convolved_data_words = list(_unencoded_convolved_data.words)
        self._unencoded_convolved_data_words += _unencoded_convolved_data_words

        _encoded_convolved_data = preprocessor.preprocess(_unencoded_convolved_data)
        assert isinstance(_encoded_convolved_data, MaskedArray)
        assert isinstance(self._encoded_convolved_data, (MaskedArray, type(None)))

        self._encoded_convolved_data = append_masked(self._encoded_convolved_data, _encoded_convolved_data, axis=0) if self._encoded_convolved_data is not None else _encoded_convolved_data


    @property
    def original_length(self):
        return len(self._unencoded_convolved_data_words)

    def __getitem__(self, i):
        return self._data[i]

    def get_decoded(self, i_in_convolved: int):
        return self._unencoded_convolved_data_words[i_in_convolved]

    def decode(self, t):
        raise ValueError('not implemented')

    def get_original_data_indices(self, i_in_convolved: int) -> Tuple[int, int, int]:
        """:return: see SwipeEmbeddingDataFrame.convolve"""
        return self._inverse_indices[i_in_convolved]

    def __len__(self):
        """ Gets the length of the encoded validation data. """
        return len(self._encoded_convolved_data)

    @property
    def X(self):
        """Gets the raw data."""
        return self._encoded_convolved_data


class Metrics(Callback):
    def __init__(self, 
                 validation_data: ValidationData, 
                 write_scalar: Callable[[str, int, Union[int, float]], None],  # this must be provided from outside because it must be the global writer (because multiple threads/summary writers isn't supported)
                 monitor_namespace: str = "test/",
                 print_loss=False,
                 print_misinterpretation_examples=False,
                 ):
        assert isinstance(validation_data, ValidationData)
        super().__init__()
        self.monitor_namespace = monitor_namespace
        self._write_scalar = write_scalar
        self.test_data = validation_data
        self._print_loss = print_loss
        self._print_misinterpretation_examples = print_misinterpretation_examples
        self.losses: List[float] = []

    @override
    def on_train_begin(self, logs={}):
        self._data = []
        self.losses.clear()

    @override
    def on_train_end(self, logs={}):
        self.print_misinterpreted_words()

    @override
    def on_epoch_end(self, batch, logs={}):
        if (batch % 3) == 0:
            places, occurrences, y_predict = self._get_places()

            self._write_and_print_summaries(places, occurrences, y_predict, batch, logs)

    def print_misinterpreted_words(self):
        if not self._print_misinterpretation_examples:
            return

        places, occurrences, y_predict = self._get_places()

        failed_indices = [i for place, occurrence, i in zip(places, occurrences, range(len(places))) 
                          if len(place) != 0 and occurrence != 1]

        print(f"\nTotal misinterpreted words: {len(failed_indices)}/{len(places)}{'. Examples:' if len(failed_indices) != 0 else ''}")

        for failed_index in islice(failed_indices, 10):
            place = places[failed_index]
            assert len(place) != 0, 'bug'

            swiped_word = self.test_data.get_decoded(failed_index)
            interpreted_as = ", ".join("'" + self.test_data.get_decoded(interpreted_as_index) + "'" 
                                       for interpreted_as_index in place)

            print(f"Swiping '{swiped_word}' was interpreted as {interpreted_as}")


    def _get_places(self):
        y_predict = np.asarray(self.model.predict(self.test_data.X))

        occurrences = np.zeros(self.test_data.original_length, int)
        places = [[] for _ in range(self.test_data.original_length)]
        for i in range(len(y_predict)):
            # what pieces of data are currently here:                                  related variables:
            # - a swipe                                                                swiped_id, swiped_index
            # - the word that was swiped (ground truth)                                word_id, swiped_word
            # - another swipe with which we're doing a convolution prediction          i
            # - the ground truth word for the other swipe                              input_word

            # we count the number word that predicted stronger than the ground truth prediction
            # this is the complicated part (reversed from expectation):
            # we associate that with the convolved swipe (not the ground truth swipe)

            # The new complication is duplication: word can now occur multiple times
            # so comparison by indices isn't sufficient, we have to compare by actual value


            # note that swiped_id and word_id are in the original data set
            # and swiped_index is in the convolved data set 
            # is is already the word_index, i.e. in the index in the convolved data set of the word that it was convolved with (i.e. the one it wants a prediction on, not the swiped word)
            swiped_id, word_id, swiped_index = self.test_data.get_original_data_indices(i)
            occurrences[swiped_id] += 1

            # the word that was swiped is:
            swiped_word = self.test_data.get_decoded(swiped_index)
            # the word it was convolved with:
            input_word = self.test_data.get_decoded(i)

            is_correct = (swiped_word == input_word)

            if not is_correct and y_predict[i] >= y_predict[swiped_index]:
                places[swiped_id].append(i)


        # NOTE: since duplication is allowed, I think occurrences is not correct anymore
        return places, occurrences, y_predict 


    def _write_and_print_summaries(self, places, occurrences, y_predict, batch, logs) -> float:
        s = ', '.join(islice((f"{len(place)}/{_count}" for place, _count in zip(places, occurrences)), 20))
        test_loss = sum(len(p) for p in places) / len(self.test_data)
        # score = sum(a / b for a, b in zip(place, occurrences))
        if self._print_loss:
            print(f" - test_loss: {test_loss:.3g}")
            print('\n - places: [' + s + ']')

        pred_min, pred_max = float(y_predict.min()), float(y_predict.max())

        logs[self.monitor_namespace + 'min'] = pred_min
        logs[self.monitor_namespace + 'max'] = pred_max
        logs[self.monitor_namespace + 'test_loss'] = test_loss

        self.losses.append(test_loss)

        self._write_scalar("pred_min", batch, pred_min, "test_extrema")
        self._write_scalar("pred_max", batch, pred_max, "test_extrema")
        self._write_scalar("test_loss", batch, test_loss, "test_loss")
        
        return test_loss


    def get_data(self):
        return self._data
