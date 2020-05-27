import dataclasses
from dataclasses import dataclass, MISSING
import inspect


def unordereddataclass(*args, **kwargs):
    """
    A dataclass with the small modification that all mandatory fields are placed before all optional fields (in the __init__ arg list)
    to prevent the pesky 'non-default argument {f.name!r} follows default argument' 
    """
    _original_init_fn = dataclasses._init_fn

    def _has_default(f):
        return not(f.default is MISSING and f.default_factory is MISSING)

    def _init_fn_override(fields, *args):
        return _original_init_fn(list(sorted(fields, key=_has_default)), *args)


    dataclasses._init_fn = _init_fn_override
    try:
        return dataclass(*args, **kwargs)
    finally:
        dataclasses._init_fn = _original_init_fn




def mydataclass(cls, *args, **kwargs):
    """
    A dataclass that is both unordered, and enables lazy field computation,
    i.e. you can override (in the inheritance sense) fields by properties to enable lazy computation of that field. 
    This invalidates passing such fields to __init__.

    This probably does not support inherited properties (TODO).
    """
    _original_init_fn = dataclasses._init_fn

    def cls_has_property_with_name(property_name) -> bool:
        properties = inspect.getmembers(cls, lambda o: isinstance(o, property))

        # p[0] should be read as `property.name`
        return next((p for p in properties if p[0] == property_name), None) is not None

    def _init_fn_override(fields, *args):
        fields = list(f for f in fields if not cls_has_property_with_name(f))
        return _original_init_fn(fields, *args)




    dataclasses._init_fn = _init_fn_override
    try:
        return unordereddataclass(cls, *args, **kwargs)
    finally:
        dataclasses._init_fn = _original_init_fn
