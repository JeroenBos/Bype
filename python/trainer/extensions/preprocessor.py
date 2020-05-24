from dataclasses import dataclass
from typing import Any, Tuple, Union

from trainer._trainer import TrainerExtension, Trainer, X, Y
from trainer.extensions.ComputeValueExtension import ComputeValueTrainerExtension
from utilities import override
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._2_transform import Preprocessor
from keyboard._4a_word_input_model import WordStrategy

@dataclass
class Params:
    word_input_strategy: WordStrategy
    max_timesteps: int
    convolution_fraction: Union[int, float] = 1.0


class PreprocessorTrainerExtension(TrainerExtension):
    def __init__(self, params: Params):
        self.params = params

        self.params.preprocessor = self.create_preprocessor

    def create_preprocessor(self):
        return Preprocessor(max_timesteps=self.params.max_timesteps,
                            convolution_fraction=self.params.convolution_fraction,
                            word_input_strategy=self.params.word_input_strategy,
                            )
                            
    @override
    def before_fit(self, x: SwipeEmbeddingDataFrame, y: Y) -> Tuple[X, Y]:
        assert isinstance(x, SwipeEmbeddingDataFrame), "GenerateData should have created a SwipeEmbeddingDataFrame"

        result = self.params.preprocessor.preprocess(x)
        return result, y


class SetMaxTimestepTrainerExtension(ComputeValueTrainerExtension):
    @property
    def param_name(self):
        return 'max_timesteps'

    def compute(self):
        return self.params.n_chars

class ComputeMaxTimestepTrainerExtension(ComputeValueTrainerExtension):
    @override
    @property
    def compute_on_before_compile(self) -> bool:
        return False


    def before_fit(self, x: SwipeEmbeddingDataFrame, y: None) -> Tuple[X, Y]:
        assert isinstance(x, SwipeEmbeddingDataFrame), "GenerateData should have created a SwipeEmbeddingDataFrame"

        if not hasattr(self.params, 'max_timesteps') or self.params.max_timesteps is None:
            from keyboard._1a_generate import get_timesteps
            max_timestep = len(get_timesteps(x))
            setattr(self.params, 'max_timesteps', max_timestep)

        return x, y


class ComputeSwipeFeatureCountTrainerExtension(ComputeValueTrainerExtension):
    @property
    def param_name(self):
        return 'swipe_feature_count'

    def compute(self):
        return 3 + self.params.word_input_strategy.get_feature_count()
