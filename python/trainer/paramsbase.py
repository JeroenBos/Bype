from abc import ABC
from dataclasses import dataclass
from typing import Callable

from trainer.ModelAdapter import CompileArgs, FitArgs


@dataclass
class ParamsBase(ABC):   
    fit_args: FitArgs = FitArgs()
    compile_args: CompileArgs = CompileArgs()

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if not name.startswith('__') and isinstance(value, Callable):
            return value()
        return value
