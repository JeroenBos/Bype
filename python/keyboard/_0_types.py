from typing import Dict, List, Callable, TypeVar

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
