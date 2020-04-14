from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import single_letter_swipes, double_letter_swipes  # noqa
from keyboard._2_transform import Preprocessor
from keyboard._3_model import KeyboardEstimator
from keyboard._5_output import KeyboardResultWriter
from typing import List, Union


data = SwipeEmbeddingDataFrame.__as__(double_letter_swipes).convolve()

preprocessor = Preprocessor(time_steps=len(data.swipes[0]))

training = KeyboardEstimator[preprocessor].create_initialized().fit(data)
