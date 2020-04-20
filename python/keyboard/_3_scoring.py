import math
from itertools import islice
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
from MyBaseEstimator import get_log_dir


metrics_writer = tf.summary.create_file_writer(get_log_dir('logs/') + '/metrics')

class Metrics(Callback):
    def __init__(self, preprocessed_convolved_validation_data, decode: Callable, get_original_swipe_index, L, model=None):
        super().__init__()
        self.test_data = preprocessed_convolved_validation_data
        self.decode = decode
        self.decoded_test_data = [decode(t) for t in self.test_data]
        self.get_original_swipe_index = get_original_swipe_index
        self._L = L
        if model is not None:
            self.model = model

    def on_train_begin(self, logs={}):
        self._data = []

    def on_train_end(self, logs={}):
        self.print_misinterpreted_words()

        super().on_train_end()

    def on_epoch_end(self, batch, logs={}):
        places, occurrences, y_predict = self._get_places()
        
        self._write_and_print_summaries(places, occurrences, y_predict)

        super().on_epoch_end()



    def print_misinterpreted_words(self):
        places, occurrences, y_predict = self._get_places()

        failed_indices = [i for place, occurrence, i in zip(places, occurrences, range(len(places))) 
                          if len(place) != 0 and occurrence != 1]

        print(f"\nTotal misinterpreted words: {len(failed_indices)}{'. Examples:' if len(failed_indices) != 0 else ''}")

        for failed_index in islice(failed_indices, 10):
            place = places[failed_index]
            assert len(place) != 0, 'bug'

            swiped_word = self.decode(self.test_data[failed_index])
            interpreted_as = ", ".join("'" + self.decode(self.test_data[interpreted_as_index]) + "'" 
                                       for interpreted_as_index in place)

            print(f"Swiping '{swiped_word}' was interpreted as {interpreted_as}")


    def _get_places(self):
        y_predict = np.asarray(self.model.predict(self.test_data))

        occurrences = np.zeros(self._L, int)
        places = [[] for _ in range(self._L)]
        for i in range(len(y_predict)):
            # note that swiped_id and word_id are in the original data set
            # and swiped_index is in the convolved data set 
            # is is already the word_index, i.e. in the index in the convolved data set of the word that it was convolved with (i.e. the one it wants a prediction on, not the swiped word)
            swiped_id, word_id, swiped_index = self.get_original_swipe_index.__func__(i)
            occurrences[swiped_id] += 1

            # whether it's one of the correct indices:
            is_correct = (swiped_id == word_id)

            if not is_correct and y_predict[i] >= y_predict[swiped_index]:
                places[swiped_id].append(i)

            # the word that was swiped is:
            swiped_word = self.decode(self.test_data[swiped_index])
            # the word it was convolved with:
            input_word = self.decode(self.test_data[i])

            # they cannot be the same unless the i is one of the correct indices (indicated by correct_swipe_index == correct_word_index)
            assert swiped_word != input_word or is_correct

        return places, occurrences, y_predict



    def _write_and_print_summaries(self, places, occurrences, y_predict):
        s = ', '.join(islice((f"{len(place)}/{_count}" for place, _count in zip(places, occurrences)), 20))
        test_loss = sum(len(p) for p in places) / len(self.test_data)
        # score = sum(a / b for a, b in zip(place, occurrences))
        print(f" - test_loss: {test_loss:.3g}")
        print('\n - places: [' + s + ']')

        with metrics_writer.as_default():
            tf.summary.scalar('pred/min', data=float(y_predict.min()), step=batch)
            tf.summary.scalar('pred/max', data=float(y_predict.max()), step=batch)
            tf.summary.scalar('pred/test_loss', data=test_loss, step=batch)
            metrics_writer.flush()


    def get_data(self):
        return self._data
