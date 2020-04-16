from time import time
from DataSource import InMemoryDataSource
import pandas as pd
import numpy as np
from keyboard._0_types import myNaN, Key, Keyboard, SwipeDataFrame, Input, RawTouchEvent, ProcessedInput, ProcessedInputSeries, SwipeEmbeddingDataFrame
from keyboard._1_import import raw_data, keyboard_layouts, KEYBOARD_LAYOUT_SPEC
from keyboard._3a_word_input_model import WordStrategy, CappedWordStrategy
from collections import namedtuple
from typing import Dict, List, Union, TypeVar, Callable, Tuple, Any
from utilities import print_fully
from more_itertools.more import first

def get_code(char: str) -> int:
    assert isinstance(char, str)
    assert len(char) == 1

    return ord(char[0])


def get_keyboard(keyboard_layout: Union[int, pd.DataFrame]) -> Dict[int, Key]:
    result = List[Key]
    keyboard = keyboard_layouts[keyboard_layout] if isinstance(keyboard_layout, int) else keyboard_layout

    def key_from_row(row, code, code_index):
        return Key(code, code_index, **{f'{col}': row[col] for col in keyboard.columns.values if col != 'codes'})

    # allcodes = list(code for _index, row in keyboard.iterrows() for code in row['codes'])
    # if len(allcodes) != len(set(allcodes)):
    #     raise KeyError("The same code at multiple places isn't supported yet")

    result: Dict[int, Key] = {code: key_from_row(row, code, code_index)
                              for _index, row in keyboard.iterrows()
                              for code_index, code in enumerate(row['codes'])}

    layout_id = keyboard_layout if isinstance(keyboard_layout, int) else '?'

    # infer keyboard width and height:
    lefts = (key.x for key in result.values())
    rights = (key.x + key.width for key in result.values())
    tops = (key.y for key in result.values())
    bottoms = (key.y + key.height for key in result.values())
    left = min(lefts)
    right = max(rights)
    top = min(tops)
    bottom = max(bottoms)
    width = right - left
    height = bottom - top

    return Keyboard(layout_id, width, height, left, top, iterable=result)


keyboards: List[Keyboard] = [get_keyboard(layout_index) for layout_index in range(len(keyboard_layouts))]




class Preprocessor:
    def __init__(self, 
                 max_time_steps=1,
                 word_input_strategy: WordStrategy = CappedWordStrategy(5),
                 loss_ctor='binary_crossentropy'):
        self.swipe_feature_count = 3 + word_input_strategy.get_feature_count()
        self.max_timesteps = max_time_steps
        self.batch_count = 1
        self.word_input_strategy = word_input_strategy
        self.loss_ctor = loss_ctor

    def set_params(self, **params):
        assert all(key in self.__dict__ for key in params.keys())
        self.__dict__.update(params)


    def _get_keyboard(self, touchevent: RawTouchEvent) -> Keyboard:
        assert hasattr(touchevent, "KeyboardLayout")
        assert isinstance(touchevent.KeyboardLayout, (int, np.int32, np.int64))

        layout_id = touchevent.KeyboardLayout
        keyboard = keyboards[layout_id]
        return keyboard


    def _get_normalized_x(self, touchevent: RawTouchEvent) -> float:
        return self._get_keyboard(touchevent).normalize_x(touchevent.X)


    def _get_normalized_y(self, touchevent: RawTouchEvent) -> float:
        return self._get_keyboard(touchevent).normalize_y(touchevent.Y)


    def _get_normalized_word_length(self, word: str) -> Callable[[RawTouchEvent], float]:
        if isinstance(self.word_input_strategy, CappedWordStrategy):
            return lambda ev: len(word) / self.word_input_strategy.n
        raise ValueError(f"Not implemented for word strategy + '{type(self.word_input_strategy)}'")


    def _get_key(self, char: str, touchevent: RawTouchEvent) -> Key:
        code = get_code(char)  # noqa
        keyboard = self._get_keyboard(touchevent)
        if code not in keyboard:
            return Key.NO_KEY
        else:
            key = keyboard[code]
            assert isinstance(key, Key), "Maybe later a list of keys can be implemented?"
            return key

    def _get_normalized_button_x(self, word: str, index: int) -> Callable[[RawTouchEvent], float]:
        def impl(touchevent: RawTouchEvent) -> float:
            if index >= len(word):
                return -1

            key: Key = self._get_key(word[index], touchevent)
            return key.keyboard.normalize_x(key.x + key.width / 2)
        return impl

    def _get_normalized_button_y(self, word: str, index: int) -> Callable[[RawTouchEvent], float]:
        def impl(touchevent: RawTouchEvent) -> float:
            if index >= len(word):
                return -1

            key: Key = self._get_key(word[index], touchevent)
            return key.keyboard.normalize_y(key.y + key.height / 2)
        return impl



    def preprocess(self, X: SwipeEmbeddingDataFrame) -> np.ndarray:
        # X[word][touchevent][toucheventprop]
        timesteps = set(len(swipe) for swipe in X.swipes)

        processed = self._preprocess(X)
        # processed[word, timestep][feature]

        intermediate = processed.to_numpy()
        # intermediate has shape ndarray[word, timestep]List[feature] which isn't much better than processed tbh


        shape = [len(processed), max(timesteps), self.swipe_feature_count]
        result = np.empty(shape, dtype=np.float)
        val = result[0, 0, 0]  # noqa
        for w in range(len(processed)):
            for t in range(self.max_timesteps):
                for f in range(self.swipe_feature_count):
                    result[w, t, f] = intermediate[w, t][f]

        # max([len(x) for x in [a, b]])
        # np.concatenate([np.zeros(len(b), dtype=bool), np.ones(max_entries - len(b), dtype=bool)])
        return np.ma.masked_invalid(result)

    def _preprocess(self, X: SwipeEmbeddingDataFrame) -> ProcessedInputSeries:
        assert SwipeEmbeddingDataFrame.is_instance(X)
        assert isinstance(X.swipes, pd.Series)
        assert isinstance(X.words, pd.Series)

        if isinstance(self.word_input_strategy, CappedWordStrategy):
            # this means the word input is appended to every timestep in the swipe data
            starttime = time()
            this = self

            def f(x):
                return this.encode_padded(x.swipes, x.words)

            result = X.apply(axis=1, func=f, result_type='expand')
            print(f'applying encoding took {time() - starttime} seconds')
            return result
        else:
            raise ValueError()

    def encode_padded(self, swipe: SwipeDataFrame, word: str) -> ProcessedInput:
        """ Pads the encoding with nanned-out timesteps until self.max_timesteps. """
        encoded = self.encode(swipe, word)
        while(len(encoded)) < self.max_timesteps:
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


        features_per_time_step: List[Callable[[Dict[str, Any]], float]] = [
            self._get_normalized_x,
            self._get_normalized_y,
            self._get_normalized_word_length(word),
        ]
        if isinstance(self.word_input_strategy, CappedWordStrategy):
            for i in range(self.word_input_strategy.n):
                features_per_time_step.append(self._get_normalized_button_x(word, i))
                features_per_time_step.append(self._get_normalized_button_y(word, i))

        result: List[List[float]] = []
        for touchevent in swipe.rows():
            time_step = []
            for feature_per_time_step in features_per_time_step:
                time_step.append(feature_per_time_step(touchevent))
            result.append(time_step)

        assert self.swipe_feature_count == len(features_per_time_step)
        assert self.max_timesteps >= len(result)
        return result

    def decode(self, x: ProcessedInput) -> Input:
        """Converts the specified list of features into a word and swipe."""
        return Input('', '')

    def save(self, filepath: str) -> None:
        raise ValueError('not implemented')
