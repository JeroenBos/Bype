from typing import Dict, List, Callable, TypeVar, Any, Tuple, Optional
import pandas as pd
from abc import ABC

T = TypeVar('T')


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

    def normalize_x(self, x: int) -> float:
        return (x - self.left) / self.width

    def normalize_y(self, y: int) -> float:
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



class SwipeEmbeddingDataFrame(pd.DataFrame):
    """
    Represents a collection of swipes.

    This class serves no purposes other than type hints.
    """

    swipes: SwipeDataFrame
    words: pd.Series

    columns: Any

    @staticmethod 
    def is_instance(obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame) and sorted(obj.columns.values) == sorted(['swipes', 'words']) 



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
