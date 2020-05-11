import pandas as pd
import numpy as np
from typing import Any, Callable, List, Union, Dict, Type, Iterable, Tuple
from functools import lru_cache
import inspect 
from pathlib import Path
import os

def print_fully(df: pd.DataFrame) -> None:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def create_empty_df(length: int, columns: Union[List[str], Dict[str, Type]], verify=False, **defaults: Union[Any, Callable[[], Any]]) -> pd.DataFrame:
    """ Creates an empty df of the correct format and shape, initialized with default values, which default to 0. """
    for key, value in defaults.items():
        if key not in set(columns):
            raise ValueError(f"Unexpected default value specified for '{str(key)}'")

    # get lazily evaluable defaults
    defaults = {key: (value() if isinstance(value, Callable) else value) for key, value in defaults.items()}

    # pad defaults with defaults
    if isinstance(columns, List):  # TODO: change signature to disallow specifying columns as list
        for column in columns:
            if column not in defaults:
                defaults[column] = 0
    else:
        for column, dtype in columns.items():
            if column not in defaults:
                if dtype is bool:
                    defaults[column] = False
                else:
                    try:
                        defaults[column] = (dtype)(0)
                    except (TypeError, ValueError):
                        raise ValueError(f"The specified dtype '{str(dtype)}' has unknown default. Specify it manually")


    result = pd.DataFrame([list(defaults[key] for key in columns) for _ in range(length)], columns=columns)
    if isinstance(columns, Dict):
        result.astype(dtype=columns, copy=False)
    if verify:
        if isinstance(columns, Dict):
            def validate(self: pd.DataFrame):
                for column in self.columns:
                    if column not in set(columns):
                        raise ValueError(f"A wild column appeared: '{str(column)}'")
                for column, dtype in columns.items():
                    for elem in result[column]:
                        assert isinstance(elem, dtype)
        else:
            def validate(self: pd.DataFrame):
                for column in self.columns:
                    if column not in columns:
                        raise ValueError(f"A wild column appeared: '{str(column)}'")
                for column, dtype in zip(self.columns, self.dtypes):
                    if dtype.type is not np.object_:
                        for elem in result[column]:
                            assert isinstance(elem, dtype.type)
        bind(result, validate)
        result.validate()
    return result


def memoize(decorate_function: Callable):
    """ 
    This decorator memoizes the results of the specified function.
    """
    return lru_cache(maxsize=None)(decorate_function)


def print_repr_on_call(decorate_function: Callable):
    """ 
    This decorator prints the name of the function upon its call. 
    """
    def decorating_function(*args, **kwargs):
        arg_reprs = [repr(arg) for arg in args] + [k + '=' + repr(v) for k, v in kwargs.items()]
        print(f'---------:{decorate_function.__name__}({", ".join(arg_reprs)})')
        return decorate_function(*args, **kwargs)
    return decorating_function


def get_declaring_class(method):
    """ Gets the class that declared the specified method """
    # copied from https://stackoverflow.com/a/25959545/308451

    if inspect.isfunction(method):
        return getattr(inspect.getmodule(method), method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])

    if inspect.ismethod(method):
        print('this is a method')
        for cls in inspect.getmro(method.__self__.__class__):
            if cls.__dict__.get(method.__name__) is method:
                return cls
        raise ValueError('Specified method could not be found in its mro chain')
    raise ValueError('Specified method is not a method')

def first_non_whitespace_char_is_any_of(s: str, *chars: str) -> bool:
    assert isinstance(s, str)

    chars = set(chars)
    for c in s:
        if not c.isspace():
            return c in chars

    # in thise case 's' is whitespace only
    return False


def incremental_paths(path_format: str):
    """ Yields all paths incrementally until one doesn't exist 
    :param format: must contain %d
    """
    assert isinstance(path_format, str)
    assert '%d' in path_format

    i = 0
    while True:
        path = path_format % i
        if(not Path(path).exists()):
            break
        yield path
        i = i + 1

def read_all(path: str) -> str:
    with open(path) as file:
        return file.read()

def split_by(s: str, *separators: str) -> List[str]:
    """ Splits a string by many separators. """
    assert isinstance(separators, (List, Tuple))
    assert all(isinstance(sep, str) for sep in separators)

    if len(separators) == 0:
        return [s]
    if len(separators) == 1:
        return s.split(separators[0])

    segments = s.split(separators[0])
    result = []
    for segment in segments:
        result.extend(split_by(segment, *separators[1:]))
    return result

class Sentinel:
    def __init__(self, repr):
        self.repr = repr

    def __repr__(self):
        return f"<sentinel object '{self.repr}'>"


SKIP = Sentinel('SKIP')
def windowed_2(seq, start_pad=SKIP, end_pad=SKIP) -> Iterable[Tuple[Any, Any]]:
    prev = start_pad
    for elem in seq:
        if prev is not SKIP:
            yield prev, elem
        prev = elem
    if prev is not SKIP and end_pad is not SKIP:
        yield prev, end_pad

def is_list_of(list, element_type) -> bool:
    return isinstance(list, (List, Tuple)) and all(isinstance(elem, element_type) for elem in list)

def split_at(s: str, *indices: int) -> List[str]:
    assert is_list_of(indices, int)
    return [s[a:b] for a, b in windowed_2(sorted(indices), 0, len(s))]

def concat(list_of_lists: Iterable) -> list:
    from itertools import chain
    return list(chain.from_iterable(list_of_lists))



def get_resource(file_name):
    return os.path.join('/home/jeroen/git/bype/python/data', file_name)

def skip(iterable, n):
    iterator = iter(iterable)
    for i in range(n):
        next(iterator)
    return iterator

def read_json(path) -> Any:
    import json
    with open(path, 'r') as f:
        return json.loads(f.read())


def read_json_to_string(path) -> Any:
    import json
    with open(path, 'r') as f:
        return json.loads(f.read())


def interpolate(a, b, f): 
    def _impl(_a, _b, _f):
        return _a + (_b - _a) * _f

    if isinstance(a, (int, float)):
        return _impl(a, b, f)

    return tuple(interpolate(*t, f) for t in zip(a, b))
