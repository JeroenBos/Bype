from typing import Type, Dict, List, Callable, TypeVar, Any, Tuple, Optional, Union
import pandas as pd
from abc import ABC
import numpy as np
from python.model_training import DataSource

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

    NO_KEY: "Key"


Key.NO_KEY = Key(code=0, code_index=0, x=0, y=0, width=0, height=0,
                 edgeFlags=0, repeatable=False, toggleable=False, keyboard=None)



class SwipeDataFrame(pd.DataFrame):
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
    KeyboardWidth: pd.Series
    KeyboardHeight: pd.Series

    @property
    def columns(self):
        return super().columns


    # TODO: implement like https://stackoverflow.com/q/13135712/308451
    @staticmethod 
    def is_instance(obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame) and 'XPrecision' in obj.columns.values



class SwipeEmbeddingDataFrame(pd.DataFrame, DataSource):
    """
    Represents a collection of swipes and associated words.

    This class serves no purposes other than type hints.
    """

    swipes: pd.Series  # with elements of type SwipeDataFrame, but I can't specify that here
    words: pd.Series

    columns: Any

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        assert SwipeEmbeddingDataFrame.is_instance(self)
        assert len(self.words) == len(self.swipes), f"Incommensurate lists of swipes and words given"
        assert all(SwipeDataFrame.is_instance(swipe) for i, swipe in self.swipes.iteritems()), 'Not all specified swipes are SwipeDataFrames'
        assert all(isinstance(word, str) for i, word in self.words.iteritems()), 'Not all specified words are strings'
        
        

    @staticmethod 
    def is_instance(obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame) and sorted(obj.columns.values) == sorted(['swipes', 'words']) 


    @staticmethod
    def __as__(embedding_dataframe: pd.DataFrame) -> "SwipeEmbeddingDataFrame":
        if isinstance(embedding_dataframe, SwipeEmbeddingDataFrame):
            return embedding_dataframe
        else:
            return SwipeEmbeddingDataFrame(embedding_dataframe)

    def get_train(self):
        return self

    def get_target(self):
        return self.words




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
    KeyboardWidth: int
    KeyboardHeight: int

    SPEC: Dict[str, Type] = {
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
               "KeyboardWidth": np.int32,
               "KeyboardHeight": np.int32,
            }

    @staticmethod
    def get_type(field: str) -> Type:
        return RawTouchEvent.SPEC[field]


    @staticmethod
    def get_keys() -> List[str]:
        return list(RawTouchEvent.SPEC.keys())
