from abc import ABC


class WordStrategy(ABC):
    """
    This specs a subpart of the model, namely the input of words.
    Specifies that the words Input layer is of the specified fixed length.
    """
    pass


class CappedWordStrategy(WordStrategy):
    def __init__(self, n: int):
        """:param n: The first n characters in the word will be used. """
        assert isinstance(n, int)
        assert n > 0
        self.n = n
