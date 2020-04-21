import math
from itertools import islice
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import Callback  # noqa
from sklearn.metrics import roc_auc_score
from keyboard._0_types import SwipeEmbeddingDataFrame, Input, ProcessedInput, SwipeDataFrame, SwipeConvolutionDataFrame
from typing import List, Callable, Tuple
from collections import namedtuple
from itertools import count, takewhile
from tensorflow.keras.losses import Loss  # noqa
from keyboard._2_transform import Preprocessor
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution  # noqa
from tensorflow.python.framework import ops  # noqa
from tensorflow.python.ops import math_ops  # noqa
from tensorflow.python.keras import backend as K  # noqa
from MyBaseEstimator import get_log_dir


metrics_writer = tf.summary.create_file_writer(get_log_dir('logs/') + '/metrics')

class ValidationData:
    def __init__(self, unconvolved_data: SwipeEmbeddingDataFrame, preprocessor: Preprocessor):
        self.original_length = len(unconvolved_data)
        self._inverse_indices = []
        self._unencoded_convolved_data = unconvolved_data.convolve(preprocessor.convolution_fraction,
                                                                   inverse_indices_out=self._inverse_indices)
        self._encoded_convolved_data = preprocessor.preprocess(self._unencoded_convolved_data)
        assert len(self._inverse_indices) == len(self._unencoded_convolved_data)
        assert self._encoded_convolved_data.shape[0] == len(self._unencoded_convolved_data)

    def __getitem__(self, i):
        return self._data[i]

    def get_decoded(self, i_in_convolved: int):
        return self._unencoded_convolved_data.words[i]

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
    def __init__(self, validation_data: ValidationData, model=None):
        super().__init__()
        self.test_data = validation_data
        if model is not None or not hasattr(self, 'model'):
            self.model = model
        self.losses: List[float] = []

    def on_train_begin(self, logs={}):
        self._data = []
        self.losses.clear()

    def on_train_end(self, logs={}):
        self.print_misinterpreted_words()

    def on_epoch_end(self, batch, logs={}):
        if (batch % 10) == 0:
            places, occurrences, y_predict = self._get_places()

            self._write_and_print_summaries(places, occurrences, y_predict, batch)

    def print_misinterpreted_words(self):
        places, occurrences, y_predict = self._get_places()

        failed_indices = [i for place, occurrence, i in zip(places, occurrences, range(len(places))) 
                          if len(place) != 0 and occurrence != 1]

        print(f"\nTotal misinterpreted words: {len(failed_indices)}{'. Examples:' if len(failed_indices) != 0 else ''}")

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
            # note that swiped_id and word_id are in the original data set
            # and swiped_index is in the convolved data set 
            # is is already the word_index, i.e. in the index in the convolved data set of the word that it was convolved with (i.e. the one it wants a prediction on, not the swiped word)
            swiped_id, word_id, swiped_index = self.test_data.get_original_data_indices(i)
            occurrences[swiped_id] += 1

            # whether it's one of the correct indices:
            is_correct = (swiped_id == word_id)

            if not is_correct and y_predict[i] >= y_predict[swiped_index]:
                places[swiped_id].append(i)

            # the word that was swiped is:
            swiped_word = self.test_data.get_decoded(swiped_index)
            # the word it was convolved with:
            input_word = self.test_data.get_decoded(i)

            # they cannot be the same unless the i is one of the correct indices (indicated by correct_swipe_index == correct_word_index)
            assert swiped_word != input_word or is_correct

        return places, occurrences, y_predict


    def _write_and_print_summaries(self, places, occurrences, y_predict, batch, logs={}) -> float:
        s = ', '.join(islice((f"{len(place)}/{_count}" for place, _count in zip(places, occurrences)), 20))
        test_loss = sum(len(p) for p in places) / len(self.test_data)
        # score = sum(a / b for a, b in zip(place, occurrences))
        print(f" - test_loss: {test_loss:.3g}")
        print('\n - places: [' + s + ']')

        pred_min, pred_max = float(y_predict.min()), float(y_predict.max())

        logs['pred/min'] = pred_min
        logs['pred/max'] = pred_max
        logs['pred/test_loss'] = test_loss


        self.losses.append(test_loss)

        with metrics_writer.as_default():
            tf.summary.scalar('pred/min', data=pred_min, step=batch)
            tf.summary.scalar('pred/max', data=pred_max, step=batch)
            tf.summary.scalar('pred/test_loss', data=test_loss, step=batch)
            metrics_writer.flush()
        return test_loss


    def get_data(self):
        return self._data
