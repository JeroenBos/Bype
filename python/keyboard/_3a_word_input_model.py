from abc import ABC
from typing import Any
from myjson import json

class WordStrategy(ABC):
    """
    This specs a subpart of the model, namely the input of words.
    Specifies that the words Input layer is of the specified fixed length.
    """

    def get_feature_count(self) -> int:
        raise ValueError('Not implemented')


class CappedWordStrategy(WordStrategy):
    def __init__(self, n: int):
        """:param n: The first n characters in the word will be used. """
        assert isinstance(n, int)
        assert n > 0
        self.n = n

    def get_feature_count(self):
        return 2 * self.n

    @json
    def __repr__(self):
        return f"{CappedWordStrategy.__name__}(n={self.n})"
