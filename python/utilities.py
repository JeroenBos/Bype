import pandas as pd
import numpy as np
from typing import Any, Callable, List, Union, Dict, Type
from functools import lru_cache

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


def create_empty_df(length: int, columns: Union[List[str], Dict[str, Type]], **defaults: Union[Any, Callable[[], Any]]) -> pd.DataFrame:
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

