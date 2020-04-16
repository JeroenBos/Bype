from typing import Callable, Iterable, Union
from keyboard._0_types import T



class GeneratorWithLength:
    """
    A generator that knows its length in O(1).
    """

    def __init__(self, iterable_or_function: Union[Iterable, Callable[[int], T]], length: int):
        self.length = length
        self.is_iterable = isinstance(iterable_or_function, Iterable)
        self.iterable = (iterable_or_function(i) for i in range(length)) if isinstance(iterable_or_function, Callable) \
            else iterable_or_function
        assert len(list(self.iterable)) == length

    def __iter__(self):
        actual_length = 0
        for i, elem in enumerate(self.iterable):
            yield elem
            actual_length = i + 1
        assert self.length == actual_length, f"The iterable did not have the specified length (actual: {actual_length}, expected: {self.length})"

    def __len__(self):
        return self.length

    def __add__(self, other):
        if isinstance(other, _gwlClass):
            return GeneratorWithLength((*self.iterable, *other.iterable), len(self) + len(other))
        elif isinstance(other, Iterable) and hasattr(other, '__len__'):
            return GeneratorWithLength((*self.iterable, *other), len(self) + len(other))
        return super(self.__class__, self).__add__(other)


_gwlClass = GeneratorWithLength
