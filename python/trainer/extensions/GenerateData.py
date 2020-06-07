from dataclasses import dataclass
from typing import Optional, Union
from trainer.trainer import TrainerExtension
from keyboard._0_types import SwipeEmbeddingDataFrame, SwipeConvolutionDataFrame
from keyboard._1a_generate import perfect_swipes
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import ValidationData
from utilities import memoize


@dataclass
class Params:
    n_words: int
    n_chars: int
    verify: bool = False
    convolution_fraction: Optional[Union[int, float]] = 1.0

    @property
    def validation_data(self):
        return GenerateDataTrainerExtension_compute_validation_data(data=self.data(), preprocessor=self.preprocessor)

    @property
    def convolved_data(self):
        return GenerateDataTrainerExtension_compute_convolved_data(data=self.data(), verify=self.verify, fraction=self.convolution_fraction)



class GenerateDataTrainerExtension(TrainerExtension):
    def initialize(self):
        self.params.data = lambda: GenerateDataTrainerExtension_compute_data(n_words=self.params.n_words, n_chars=self.params.n_chars, verify=self.params.verify)

    def before_fit(self, x, y):
        assert x is None and y is None, "Data was already created"
        result = self.params.convolved_data
        assert set(result.columns.array) == set(['swipes', 'words', 'correct'])

        return result, None

@memoize
def GenerateDataTrainerExtension_compute_data(n_words: int, n_chars: int, verify: bool) -> SwipeEmbeddingDataFrame:
    raw_data = perfect_swipes(n_words, n_chars)
    data = SwipeEmbeddingDataFrame.__as__(raw_data, verify=verify) 
    return data

@memoize
def GenerateDataTrainerExtension_compute_validation_data(data: SwipeEmbeddingDataFrame, preprocessor: Preprocessor) -> ValidationData:
    return _unmemoized_generateDataTrainerExtension_compute_validation_data(data, preprocessor)

def _unmemoized_generateDataTrainerExtension_compute_validation_data(data: SwipeEmbeddingDataFrame, preprocessor: Preprocessor) -> ValidationData:
    return ValidationData(data, preprocessor)

@memoize
def GenerateDataTrainerExtension_compute_convolved_data(data: SwipeEmbeddingDataFrame, verify: bool, fraction: float):
    return data.convolve(fraction=1, verify=verify)
