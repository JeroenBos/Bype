from typing import Any, Callable, List, Union, Dict, Type
import pandas as pd
from ctypes import cast


def generic(*params: str) -> type:
    """
    Creates a metatype representing a class with the specified type parameters.
    Example usages:

    class GenericType(metaclass=generic('T', 'U')):
        pass

    class GenericTypeWithSuperclass(metaclass=generic('T', 'U'), superclass):
        pass

    also note that the superclass can be accessed via super(self.__class__, self) and super(cls, cls)
    """
    if len(params) == 0:
        params = ['T']
    if isinstance(params, str):
        params = [params]

    class Metatype(type):
        def __getitem__(self, *args):
            if(len(args) != len(params)):
                raise ValueError(f"{len(params)} type arguments must be specified.")
            # for arg in args:
            #     if not isinstance(arg, type):
            #         raise ValueError()

            newcls = type(f"{self.__name__}<{', '.join(params)}>", self.__bases__, dict(self.__dict__))
            for typeArg, name in zip(args, params):
                setattr(newcls, name, typeArg)
            return newcls
    return Metatype


# I'm abusing this file here to mean 'utils' rather than generics

def create_empty_df(length: int, columns: Union[List[str], Dict[str, Type]], **defaults: Union[Any, Callable[[], Any]]) -> pd.DataFrame:
    """ Creates an empty df of the correct format and shape, initialized with default values, which default to 0. """
    for key, value in defaults.items():
        if key not in set(columns):
            raise ValueError(f"Unexpected default value specified for '{str(key)}'")

    # get lazily evaluable defaults
    defaults = {key: (value() if isinstance(value, Callable) else value) for key, value in defaults.items()}

    # pad defaults with defaults
    if isinstance(columns, List):
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
    return result
