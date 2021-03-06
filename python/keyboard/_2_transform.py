from math import floor
from time import time
from DataSource import InMemoryDataSource
import pandas as pd
import numpy as np
from keyboard._0_types import myNaN, Key, Keyboard, SwipeDataFrame, Input, RawTouchEvent, ProcessedInput, ProcessedInputSeries, SwipeEmbeddingDataFrame
from keyboard._1_import import keyboards
from keyboard._4a_word_input_model import WordStrategy, CappedWordStrategy
from collections import namedtuple
from typing import Dict, List, Union, TypeVar, Callable, Tuple, Any, Optional
from utilities import print_fully
from more_itertools.more import first
import json as JSON
from myjson import json
from sklearn.base import BaseEstimator
from keyboard._2a_feature import Feature, InverseFeature





class Preprocessor:

    def __init__(self, 
                 max_timesteps=None,
                 convolution_fraction=1.0,
                 word_input_strategy: WordStrategy = CappedWordStrategy(5)
                 ):
        """
        :param max_timesteps: The maximum number of allowed timesteps. None for infinity. 
        """
        self.swipe_feature_count = 3 + word_input_strategy.feature_count
        self.max_timesteps = max_timesteps
        self.batch_count = 1
        self.word_input_strategy = word_input_strategy
        self._features_per_time_step = None
        self._inverse_features = None
        self.convolution_fraction = convolution_fraction

    @property
    def features_per_time_step(self):
        if self._features_per_time_step is None:
            self._features_per_time_step, self._inverse_features = self._compute_features()
            assert self.swipe_feature_count == len(self.features_per_time_step)
        return self._features_per_time_step


    @property
    def inverse_features(self):
        if self._inverse_features is None:
            self._features_per_time_step, self._inverse_features = self._compute_features()
        return self._inverse_features

    def _get_keyboard(self, touchevent: RawTouchEvent) -> Keyboard:
        assert hasattr(touchevent, "KeyboardLayout")
        assert isinstance(touchevent.KeyboardLayout, (int, np.int32, np.int64))

        layout_id = touchevent.KeyboardLayout
        keyboard = keyboards[layout_id]
        return keyboard


    def _get_normalized_x(self, touchevent: RawTouchEvent, word: str) -> float:
        return self._get_keyboard(touchevent).normalize_x(touchevent.X)

    def _get_normalized_y(self, touchevent: RawTouchEvent, word: str) -> float:
        return self._get_keyboard(touchevent).normalize_y(touchevent.Y)


    def _get_normalized_word_length(self, touchevent: RawTouchEvent, word: str) -> float:
        if isinstance(self.word_input_strategy, CappedWordStrategy):
            return len(word) / self.word_input_strategy.n
        raise ValueError(f"Not implemented for word strategy + '{type(self.word_input_strategy)}'")

    def _get_denormalized_word_length(self, word_length_feature: float, keyboard=None) -> int:
        if isinstance(self.word_input_strategy, CappedWordStrategy):
            return floor(self.word_input_strategy.n * word_length_feature)
        raise ValueError(f"Not implemented for word strategy + '{type(self.word_input_strategy)}'")


    def _get_key(self, char: str, touchevent: RawTouchEvent) -> Key:
        assert isinstance(char, str)
        assert len(char) == 1

        code = ord(char[0])
        keyboard = self._get_keyboard(touchevent)
        if code not in keyboard:
            return Key.NO_KEY
        else:
            key = keyboard[code]
            assert isinstance(key, Key), "Maybe later a list of keys can be implemented?"
            return key

    def _get_normalized_button_x(self, index: int) -> Feature:
        def feature(touchevent, word):
            if index >= len(word):
                return -1
            key: Key = self._get_key(word[index], touchevent)
            return key.keyboard.normalize_x(key.x + key.width / 2)
        return feature

    def _get_normalized_button_y(self, index: int) -> Feature:
        def feature(touchevent, word):
            if index >= len(word):
                return -1

            key: Key = self._get_key(word[index], touchevent)
            return key.keyboard.normalize_y(key.y + key.height / 2)
        return feature

    def _get_denormalized_button(self, x_feature: float, y_feature: float, keyboard: Keyboard) -> str:
        """
        This represents the inverse function of _get_normalized_button_x/y.
        Given the normalized x and y features of the button, gets the button on the specified keyboard from which the features originated.
        """
        if x_feature == -1:
            return -1

        x_button_middle = keyboard.denormalize_x(x_feature)
        y_button_middle = keyboard.denormalize_y(y_feature)

        def is_key(key: Key) -> bool:
            """
            Determines whether the specified x and y feature originate from the specified key 
            by whether the top-left button position in pixels is off by at most 1 in either direction
            """
            predicted_x = x_button_middle - key.width // 2
            predicted_y = y_button_middle - key.height // 2

            return abs(predicted_x - key.x) <= 1 and abs(predicted_y - key.y) <= 1

        keys = keyboard.values()  # keyboard is a dict of Keys, so yeah, this looks wrong but isnt'

        correctly_positioned_keys = [key for key in keys if is_key(key)]
        if len(correctly_positioned_keys) == 0:
            raise ValueError("Couldn't find key")
        elif len(correctly_positioned_keys) != 1:
            raise ValueError("Found multiple keys on keyboard at same location")

        return correctly_positioned_keys[0].char


    def preprocess(self, x: SwipeEmbeddingDataFrame) -> np.ndarray:
        # X[word][touchevent][toucheventprop]
        max_timestep = max(len(swipe) for swipe in x.swipes)
        if self.max_timesteps is not None and max_timestep > self.max_timesteps:
            raise ValueError('Too many timesteps')

        processed = self._preprocess(x)
        # processed[word, timestep][feature]

        intermediate = processed.to_numpy()
        # intermediate has shape ndarray[word, timestep]List[feature] which isn't much better than processed tbh


        shape = [len(processed), self.max_timesteps, self.swipe_feature_count]
        result = np.empty(shape, dtype=np.float)
        for w in range(len(processed)):
            for t in range(max_timestep):  # note that this is explicitly not self.max_timesteps; the excess is left masked
                for f in range(self.swipe_feature_count):
                    result[w, t, f] = intermediate[w, t][f]

        return np.ma.masked_invalid(result, copy=False)

    def _preprocess(self, X: SwipeEmbeddingDataFrame) -> ProcessedInputSeries:
        assert SwipeEmbeddingDataFrame.is_instance(X)
        assert isinstance(X.swipes, pd.Series)
        assert isinstance(X.words, pd.Series)

        max_timestep = max(len(swipe) for swipe in X.swipes)

        if isinstance(self.word_input_strategy, CappedWordStrategy):
            # this means the word input is appended to every timestep in the swipe data
            starttime = time()
            this = self

            def f(x: SwipeEmbeddingDataFrame):
                return this.encode_padded(x.swipes, x.words, max_timestep)

            result = X.apply(axis=1, func=f, result_type='expand')
            print(f'applying encoding took {time() - starttime} seconds')
            return result
        else:
            raise ValueError()

    def _compute_features(self):

        features_per_time_step: List[Feature] = [
            self._get_normalized_x,
            self._get_normalized_y,
            self._get_normalized_word_length,
        ]
        inverses: List[InverseFeature] = [
            InverseFeature(self._get_denormalized_word_length, features_per_time_step.index(self._get_normalized_word_length)),
        ]

        if isinstance(self.word_input_strategy, CappedWordStrategy):
            for i in range(self.word_input_strategy.n):
                features_per_time_step.append(self._get_normalized_button_x(i))
                features_per_time_step.append(self._get_normalized_button_y(i))
                inverses.append(InverseFeature(self._get_denormalized_button, len(features_per_time_step) - 2, len(features_per_time_step) - 1))
        else:
            raise ValueError('Not implemented')

        return features_per_time_step, inverses


    def encode_padded(self, swipe: SwipeDataFrame, word: str, max_timestep: int) -> ProcessedInput:
        """ Pads the encoding with nanned-out timesteps until max_timestep. """
        assert self.max_timesteps is None or self.max_timesteps >= max_timestep
        encoded = self.encode(swipe, word)
        while(len(encoded)) < max_timestep:
            encoded.append([myNaN] * len(encoded[0]))
        return encoded

    def encode(self, swipe: SwipeDataFrame, word: str) -> ProcessedInput:
        """
        Converts the specified word and swipe data into a list of features.
        :param swipe: A row in DataSource.get_train().
        :param word: A row in DataSource.get_target().
        """
        assert SwipeDataFrame.is_instance(swipe)
        assert isinstance(word, str)
        assert 'X' in swipe
        assert 'Y' in swipe

        result: List[List[float]] = []
        for touchevent in swipe.rows():
            time_step = []
            for feature_per_time_step in self.features_per_time_step:
                time_step.append(feature_per_time_step(touchevent, word))
            result.append(time_step)

        assert self.swipe_feature_count == len(self.features_per_time_step)
        assert self.max_timesteps is None or self.max_timesteps >= len(result)
        return result

    def decode(self, x: ProcessedInput) -> str:
        """
        Converts the specified list of features into (some of) its original input features.
        For now only returns the first 5 letters of the word.
        """
        features = [inverseFeature(x[0], keyboards[0]) for inverseFeature in self.inverse_features]
        assert isinstance(features[0], int)
        for i in range(1, len(features)): 
            assert (features[i] == -1) == (i > features[0])
        result = "".join(features[1:(features[0] + 1)])
        return result

    @json
    def __repr__(self):
        param_names = BaseEstimator._get_param_names.__func__(Preprocessor)
        args = ",".join(sorted(f'{key}={repr(self.__dict__[key])}' for key in param_names))
        return f"{Preprocessor.__name__}({args})"

    def save(self, filepath: str) -> None:
        representation = repr(self)

        json_object = JSON.dumps('"' + representation + '"', indent=4) 

        with open(filepath, "w") as outfile: 
            outfile.write(json_object)
