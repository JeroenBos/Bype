from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import single_letter_swipes, double_letter_swipes, single_and_double_letter_swipes  # noqa
from keyboard._2_transform import Preprocessor
from keyboard._3_model import KeyboardEstimator
from typing import List, Union

data = SwipeEmbeddingDataFrame.__as__(single_and_double_letter_swipes).convolve()

preprocessor = Preprocessor(time_steps='?')

training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=10).fit(data)
