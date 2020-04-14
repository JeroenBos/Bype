from DataSource import InMemoryDataSource
import pandas as pd
import numpy as np
from keyboard._0_types import Key, Keyboard, SwipeDataFrame, Input, RawTouchEvent, ProcessedInput, ProcessedInputSeries, SwipeEmbeddingDataFrame
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
                 time_steps=1,
                 word_input_strategy: WordStrategy = CappedWordStrategy(5),
                 loss_ctor='binary_crossentropy'):
        self.swipe_feature_count = 3 + word_input_strategy.get_feature_count()
        self.swipe_timesteps_count = time_steps
        self.batch_count = 1
        self.word_input_strategy = word_input_strategy
        self.loss_ctor = loss_ctor

    def set_params(self, **params):
        assert all(key in self.__dict__ for key in params.keys())
        self.__dict__.update(params)


    def _get_keyboard(self, touchevent: RawTouchEvent) -> Keyboard:
        assert hasattr(touchevent, "KeyboardLayout")
        print(type(touchevent.KeyboardLayout))
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
        code = get_code(char)
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

        processed = self._preprocess(X)
        # processed[word, timestep][feature]

        intermediate = processed.to_numpy()
        # intermediate has shape ndarray[word, timestep]List[feature] which isn't much better than processed tbh

        # I've been stuck on this for hours, I'll just do it myself
        shape = [len(processed), self.swipe_timesteps_count, self.swipe_feature_count]
        result = np.empty(shape, dtype=np.float)
        for w in range(len(processed)):
            for t in range(self.swipe_timesteps_count):
                for f in range(self.swipe_feature_count):
                    result[w, t, f] = intermediate[w, t][f]

        return result

    def _preprocess(self, X: SwipeEmbeddingDataFrame) -> ProcessedInputSeries:
        assert SwipeEmbeddingDataFrame.is_instance(X)
        assert isinstance(X.swipes, pd.Series)
        assert isinstance(X.words, pd.Series)

        if isinstance(self.word_input_strategy, CappedWordStrategy):
            # this means the word input is appended to every timestep in the swipe data
            result = X.apply(axis=1, func=lambda x: self.encode(x.swipes, x.words), result_type='expand')
            return result
        else:
            raise ValueError()

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

        swipe.validate()

        keyboardLayoutCol = swipe['KeyboardLayout'][0]

        print(f"when accessed via the columns the type is: {type(keyboardLayoutCol)}")
        firstrow = first(row for row in swipe.rows())
        # print(firstrow)
        keyboardLayoutRow = firstrow['KeyboardLayout']
        # columnindex = swipe.columns.values.tolist().index('KeyboardLayout')
        # keyboardLayoutRow = firstrow[columnindex]
        # print(firstrow.shape)
        print(f"when accessed via the rows the type is: {type(keyboardLayoutRow)}")
        # assert isinstance(keyboardLayout, int)
        result: List[List[float]] = []
        for touchevent in swipe.rows():
            print(type(touchevent.KeyboardLayout))
            time_step = []
            for feature_per_time_step in features_per_time_step:
                time_step.append(feature_per_time_step(touchevent))
            result.append(time_step)

        assert self.swipe_feature_count == len(features_per_time_step)
        assert self.swipe_timesteps_count == len(result)
        return result

    def decode(self, x: ProcessedInput) -> Input:
        """Converts the specified list of features into a word and swipe."""
        return Input('', '')
