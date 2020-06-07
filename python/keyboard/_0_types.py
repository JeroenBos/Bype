import random 
from math import floor
from typing import Type, Dict, List, Callable, TypeVar, Any, Tuple, Optional, Union
import pandas as pd
from abc import ABC
import numpy as np
from DataSource import DataSource
from utilities import create_empty_df, bind, groupby

# keras.Layers.Marking doesn't support math.nan, nor the infinities for that matter, so I just use a different number then
myNaN = 12345678.9
# import sys
#
# class Test:
#     def write(self, *args, **kwargs):
#         if self is not None:
#             if True:
#                 pass
# 
# 
# sys.stdout = Test() 

T = TypeVar('T')
ProcessedInputSeries = pd.Series  # where every element is a ProcessedInput
ProcessedInput = List[List[float]]


class Keyboard(Dict[int, "Key"]):
    def __init__(self, layout_id: int, width: int, height: int, left: int, top: int, iterable=None):
        super().__init__(iterable)
        self.layout_id = layout_id
        self.width = width
        self.height = height
        self.left = left
        self.top = top
        for key in self.values():
            key.keyboard = self

    def normalize_x(self, x: Union[int, float]) -> float:
        return (x - self.left) / self.width

    def normalize_y(self, y: Union[int, float]) -> float:
        return (y - self.top) / self.height

    def denormalize_x(self, normalized_x: float) -> int:
        """ Returns the x in pixels that would have resulted in the specified normalized x. """
        return floor(normalized_x * self.width + self.left)

    def denormalize_y(self, normalized_y: float) -> int:
        """ Returns the y in pixels that would have resulted in the specified normalized y. """
        return floor(normalized_y * self.height + self.top)

    def get_key(self, char: Union[str, int]) -> "Key":
        assert isinstance(char, str) or isinstance(char, int)
        assert len(char) == 1
        code = ord(char) if isinstance(char, str) else char
        return self.get(code)

    def values(self) -> List["Key"]:
        return super().values()




class Key:
    def __init__(self, code: int, code_index: int, x: int, y: int, width: int, height: int,
                 edgeFlags: int, repeatable: bool, toggleable: bool, keyboard: Keyboard = None):
        self.code = code
        self.code_index = code_index
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.edge_flags = edgeFlags
        self.repeatable = repeatable
        self.toggleable = toggleable
        self.keyboard = keyboard

    @property
    def char(self):
        return chr(self.code)


    @property
    def abs_x(self):
        """ `abs` means including keyboard positioning (i.e. not relative to keyboard) """
        return self.x + self.keyboard.left

    @property
    def abs_y(self):
        """ `abs` means including keyboard positioning (i.e. not relative to keyboard) """
        return self.y + self.keyboard.top

    @property
    def abs_center_x(self):
        """ `abs` means including keyboard positioning (i.e. not relative to keyboard) """
        return self.abs_x + self.width // 2

    @property
    def abs_center_y(self):
        """ `abs` means including keyboard positioning (i.e. not relative to keyboard) """
        return self.abs_y + self.height // 2

    NO_KEY: "Key"



Key.NO_KEY = Key(code=0, code_index=0, x=0, y=0, width=0, height=0,
                 edgeFlags=0, repeatable=False, toggleable=False, keyboard=None)


class MyDataFrame(pd.DataFrame):
    columns: Any  # or this: @property def columns(self): return super().columns

    def rows(self) -> object:
        class Row:
            def __getitem__(self, *args):
                if len(args) != 1:
                    raise ValueError('Expected one key')
                return self.__dict__[args[0]]

        value = Row()  # object doesn't implement __dict__
        for i in range(len(self)):
            for column in self.columns:
                value.__dict__[column] = self[column][i]
            yield value

    def __hash__(self):
        return id(self)


class SwipeDataFrame(MyDataFrame):
    """
    Represents the data from one swipe.

    This class serves no purposes other than type hints.
    """

    PointerIndex: pd.Series
    Action: pd.Series
    Timestamp: pd.Series
    X: pd.Series
    Y: pd.Series
    Pressure: pd.Series
    Size: pd.Series
    Orientation: pd.Series
    ToolMajor: pd.Series
    ToolMinor: pd.Series
    TouchMinor: pd.Series
    TouchMajor: pd.Series
    XPrecision: pd.Series
    YPrecision: pd.Series
    EdgeFlags: pd.Series
    KeyboardLayout: pd.Series


    # TODO: implement like https://stackoverflow.com/q/13135712/308451
    @staticmethod 
    def is_instance(obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame) and 'XPrecision' in obj.columns.values

    @staticmethod
    def create_empty(length: int, **defaults) -> "SwipeDataFrame":
        """
        Creates an empty df of the correct format and shape determined by SPEC,
        initialized with default values, which default to 0.
        """

        result: pd.DataFrame = create_empty_df(length, columns=RawTouchEvent.SPEC, **defaults)
        result.astype(dtype=RawTouchEvent.SPEC, copy=False)
        bind(result, SwipeDataFrame.rows)
        return result


    @staticmethod
    def create(inputs: List[T], selector: Callable[[T, int], "Partial[RawTouchEvent]"], verify=False) -> "SwipeDataFrame":
        """
        :param selector: A function creating a partial RawTouchEvent from the specified input and index
        """
        result = SwipeDataFrame.create_empty(len(inputs))
        for i, word in enumerate(inputs):
            partialEvents = selector(word, i)
            partialEvents = partialEvents.__dict__ if not isinstance(partialEvents, Dict) else partialEvents

            for key, value in partialEvents.items():
                if key not in result.columns.values:
                    raise ValueError(f"Unknown column '{str(key)}' for input {str(word)} at index '{str(i)}'")
                result[key][i] = value

        if verify and hasattr(result, 'validate'):
            result.validate()

        return result

class SwipeEmbeddingDataFrame(MyDataFrame, DataSource):
    """
    Represents a collection of swipes and associated words.

    This class serves no purposes other than type hints.
    """

    swipes: pd.Series   # with elements of type SwipeDataFrame, but I can't specify that here
    words: pd.Series    # series of strings
    correct: pd.Series  # series of booleans

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, verify=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        if verify:
            assert SwipeEmbeddingDataFrame.is_instance(self)
            assert len(self.words) == len(self.swipes), f"Incommensurate lists of swipes and words given"
            assert all(SwipeDataFrame.is_instance(swipe) for i, swipe in self.swipes.iteritems()), 'Not all specified swipes are SwipeDataFrames'
            assert all(len(word) == 0 or isinstance(word, str) for i, word in self.words.iteritems()), 'Not all specified words are strings'



    @staticmethod 
    def is_instance(obj: Any) -> bool:
        return isinstance(obj, SwipeEmbeddingDataFrame) or \
               isinstance(obj, pd.DataFrame) and 'swipes' in obj.columns.values and 'words' in obj.columns.values


    @staticmethod
    def __as__(embedding_dataframe: pd.DataFrame, verify=False) -> "SwipeEmbeddingDataFrame":
        if isinstance(embedding_dataframe, SwipeEmbeddingDataFrame):
            return embedding_dataframe
        else:
            return SwipeEmbeddingDataFrame(embedding_dataframe, verify=verify)


    @staticmethod
    def create_empty(length: int, verify=False) -> "SwipeEmbeddingDataFrame":
        defaults = {
            'swipes': SwipeDataFrame.create_empty(0), 
            'words': pd.Series([], dtype=np.str), 
            'correct': pd.Series([], dtype=np.bool)}
        inner = create_empty_df(length, columns=list(defaults.keys()), verify=verify, **defaults)
        return SwipeEmbeddingDataFrame(inner, verify=verify)


    @staticmethod
    def create(words: List[str], swipe_selector: Callable[[str, int], SwipeDataFrame], verify=False) -> "SwipeEmbeddingDataFrame":
        """
        Creates a swipe embedding dataframe from the specified words and a mapping function creating the swipe
        :param words: The words to create swipes for.
        :param swipe_selector: A function creating the swipe data from the word and index in the param 'words'
        """
        if not isinstance(words, List):
            words = list(words)

        result = SwipeEmbeddingDataFrame.create_empty(len(words), verify=verify)
        swipes = [swipe_selector(word, i) for i, word in enumerate(words)]

        if verify:
            for swipe in swipes:
                assert isinstance(swipe, pd.DataFrame)
                assert len(swipe) != 0, 'empty swipe'
            assert len(set(len(swipe.columns) for swipe in swipes)) == 1, 'not all swipes have same columns'

        for i, swipe in enumerate(swipes):
            result.swipes[i] = swipe
            result.words[i] = words[i]
            result.correct[i] = True
        return result

    @staticmethod
    def create_from_rows(rows: List[Tuple[str, SwipeDataFrame, bool]], verify=False) -> "SwipeEmbeddingDataFrame":

        result = SwipeEmbeddingDataFrame.create_empty(len(rows), verify=verify)
        for i, row in enumerate(rows):
            assert isinstance(row.words, str) and isinstance(row.correct, bool)
            result.words[i] = row.words
            result.swipes[i] = row.swipes
            result.correct[i] = row.correct
        return result

    def get_train(self):
        return self

    def get_target(self):
        return self.correct.transform(lambda boolean: 1.0 if boolean else 0.0)

    def get_row(self, i: int) -> "Input":
        return Input(self.swipes[i], self.words[i])

    def convolve(self, fraction=1.0, inverse_indices_out: Optional[List[Tuple[int, int, int]]] = None, verify=False) -> "SwipeConvolutionDataFrame":
        """
        :param fraction: The fraction of negatives to create. 
        :param inverse_indices_out: An out parameter. If specified, the 3-tuples inverse indices will be appended; see below. 
        """
        assert type(fraction) == float or type(fraction) == int
        assert fraction >= 0.0


        L = len(self.words)
        N = floor(L * (1 + fraction))

        def is_correct(i: int) -> bool:
            return i * L // N != ((i - 1) * L) // N

        assert sum((1 if is_correct(i) else 0) for i in range(N)) == L, 'is_correct has a bug'

        def draw_non(i: int, max: int):
            """ Draws a number from the range [0, max) / { i }. """
            drawn = random.randint(0, max - 2)
            if drawn == i:
                return max - 1
            return drawn

        # index in convolution of the correct combination of 
        indices_of_swipe_in_convolved = [i for i in range(N) if is_correct(i)]

        # 3-tuples of all indices (incl inverse)
        # index_of_swipe_in_non_convolved, index_of_word_in_non_convolved, index_of_swipe_in_convolved
        indices = [(i * L // N, 
                    ((i * L // N) if is_correct(i) else draw_non(i * L // N, L)), 
                    indices_of_swipe_in_convolved[i * L // N])
                   for i in range(N)]

        if inverse_indices_out is not None:
            inverse_indices_out.extend(indices)

        words = [self.words[t[1]] for t in indices]

        data = SwipeConvolutionDataFrame.create(words, lambda word, i: self.swipes[indices[i][0]])
        data.correct = [is_correct(i) for i in range(N)]
        bind(data, SwipeConvolutionDataFrame.convolve_data)
        return data

    def groupby(self, keySelector: Callable[[pd.Series], Union[str, int]]) -> List["SwipeEmbeddingDataFrame"]:  # the pd.Series contains swipes, words and corrects
        grouped: Dict[int, List[Tuple[str, SwipeDataFrame, bool]]] = groupby(self, keySelector)

        result = [SwipeEmbeddingDataFrame.create_from_rows(rows) for rows in grouped.values()]
        return result

    def groupby_timesteps(self) -> List["SwipeEmbeddingDataFrame"]:
        return self.groupby(get_timesteps)


def get_timesteps(d):
    return len(d.swipes)


class SwipeConvolutionDataFrame(SwipeEmbeddingDataFrame):

    def convolve_data(self) -> None:
        raise ValueError("Cannot convolve already convolved dataframe")


class Input:
    """Represents the tuple 'swipe' + 'word'."""

    def __init__(self, swipe: SwipeDataFrame, word: str):
        assert isinstance(word, str)
        assert SwipeDataFrame.is_instance(swipe)
        self.swipe = swipe
        self.word = word

    @staticmethod
    def is_instance(obj) -> bool:
        return isinstance(obj, Input) and isinstance(obj.word, str) and SwipeDataFrame.is_instance(obj.swipe)



class RawTouchEvent(pd.Series):
    """Represents a row in a SwipeDataFrame"""

    PointerIndex: int
    Action: int
    Timestamp: int  # int64
    X: float
    Y: float
    Pressure: float
    Size: float
    Orientation: float
    ToolMajor: float
    ToolMinor: float
    TouchMinor: float
    TouchMajor: float
    XPrecision: float
    YPrecision: float
    EdgeFlags: float
    KeyboardLayout: int

    SPEC_np: Dict[str, Type] = {
               "PointerIndex": np.int32,
               "Action": np.int32,
               "Timestamp": np.int64,
               "X": np.float32,
               "Y": np.float32,
               "Pressure": np.float32,
               "Size": np.float32,
               "Orientation": np.float32,
               "ToolMajor": np.float32,
               "ToolMinor": np.float32,
               "TouchMinor": np.float32,
               "TouchMajor": np.float32,
               "XPrecision": np.float32,
               "YPrecision": np.float32,
               "EdgeFlags": np.float32,
               "KeyboardLayout": np.int32,
            }
    SPEC: Dict[str, Type]

    @staticmethod
    def get_type(field: str) -> Type:
        return RawTouchEvent.SPEC[field]


    @staticmethod
    def get_keys() -> List[str]:
        return list(RawTouchEvent.SPEC.keys())


np_dtype_map = {np.int32: int, np.float32: float, np.int64: int, np.bool: bool}
RawTouchEvent.SPEC = {key: np_dtype_map[value] for key, value in RawTouchEvent.SPEC_np.items()}


class RawTouchEventActions:
    # see https://developer.android.com/reference/android/view/MotionEvent
    Down = 0
    Up = 1
    Move = 2
    Cancel = 3
