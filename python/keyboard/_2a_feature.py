from typing import Callable, Any, Union
from keyboard._0_types import Keyboard, RawTouchEvent, ProcessedInput

VariadicFloatKeyboardToAnyDelegate = Union[
    Callable[[Keyboard], Any], 
    Callable[[float, Keyboard], Any], 
    Callable[[float, float, Keyboard], Any], 
    Callable[[float, float, float, Keyboard], Any], 
    Callable[[float, float, float, float, Keyboard], Any],
]

class Feature:
    """Represents the signature of a feature (per timestep). """

    def __call__(self, touchevent: RawTouchEvent, word: str) -> float:
        raise ValueError('abstract')

class InverseFeature:
    def __init__(self, f: VariadicFloatKeyboardToAnyDelegate, *feature_indices: int):
        self.f = f
        self.feature_indices = feature_indices

    def __call__(self, timestep: ProcessedInput, keyboard: Keyboard):
        if len(timestep.shape) == 1:
            features = [timestep[i] for i in self.feature_indices]
        else:
            features = [timestep[0, i] for i in self.feature_indices]
        return self.f(keyboard=keyboard, *features)
