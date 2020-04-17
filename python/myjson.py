from typing import Any, Callable, Dict
import inspect
import json as JSON
from utilities import bind, get_declaring_class

# dictionary from type name to inverse of __repr__
decoders: Dict[str, Callable[[str], Any]] = {}

def json(method: Callable):
    """
    A decorator intended to decorate the __repr__ of a class
    """
    assert isinstance(method, Callable)

    def decoder(s: str) -> Any:
        obj = JSON.loads(s)
        # assumes that __repr__ returns a string interpretable as a python ctor call returning an identical object
        # i.e. assumes that eval ∘ repr is idempotent
        representation = obj if isinstance(obj, str) else method.__call__(Object(obj))
        result = eval(representation, method.__globals__)
        return result

    # assuming the current function is called as a decorator, caller_name will be the name of the declaring class
    calling_frame = inspect.getouterframes(inspect.currentframe(), 2)[1]
    caller_name = calling_frame[3]

    decoders[caller_name] = decoder
    return method

class Object(object):
    """ Takes a dictionary and converts it into an object. """

    def __init__(self, dictionary: Dict[str, Any]):
        self.__dict__ = dictionary
