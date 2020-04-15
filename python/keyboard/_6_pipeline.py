from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import single_letter_swipes
from keyboard._2_transform import Preprocessor
from keyboard._3_model import KeyboardEstimator
from keyboard._5_output import KeyboardResultWriter
from hp import do_hp_search
from typing import List, Union


data = SwipeEmbeddingDataFrame.__as__(single_letter_swipes).convolve()

preprocessor = Preprocessor()

training = KeyboardEstimator[preprocessor].create_initialized().fit(data)
