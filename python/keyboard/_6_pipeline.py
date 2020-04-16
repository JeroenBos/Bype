from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import single_letter_swipes, double_letter_swipes, single_and_double_letter_swipes, single_double_and_triple_letter_swipes  # noqa
from keyboard._2_transform import Preprocessor
from keyboard._3_model import KeyboardEstimator
from typing import List, Union
from time import time

verify = False

starttime = time()
data: SwipeEmbeddingDataFrame = SwipeEmbeddingDataFrame.__as__(single_double_and_triple_letter_swipes(), verify)
print(f'generating data took {time() - starttime} seconds')

starttime = time()
data = data.convolve(verify)
print(f'convolving took {time() - starttime} seconds')

preprocessor = Preprocessor(max_time_steps=3)

training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=10).fit(data)
