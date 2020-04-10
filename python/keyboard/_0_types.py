from typing import Dict, List, Callable, TypeVar, Any
import pandas as pd
from abc import ABC

T = TypeVar('T')


class Key:
    def __init__(self, code: int, code_index: int, x: int, y: int, width: int, height: int,
                 edgeFlags: int, repeatable: bool, toggleable: bool):
        self.code = code
        self.code_index = code_index
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.edge_flags = edgeFlags
        self.repeatable = repeatable
        self.toggleable = toggleable


class Keyboard(Dict[int, Key]):
    def __init__(self, layout_id: int, width: int, height: int, iterable=None):
        super().__init__(iterable)
        self.layout_id = layout_id
        self.width = width
        self.height = height


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

    columns: Any


class SwipeEmbeddingDataFrame(pd.DataFrame):
    """
    Represents a collection of swipes.

    This class serves no purposes other than type hints.
    """

    swipes: SwipeDataFrame

    columns: Any
