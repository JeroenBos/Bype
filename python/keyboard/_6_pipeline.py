import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import single_letter_swipes, double_letter_swipes, single_and_double_letter_swipes, single_double_and_triple_letter_swipes  # noqa
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics
from keyboard._4_model import KeyboardEstimator
from typing import List, Union
from time import time

verify = False

starttime = time()
data: SwipeEmbeddingDataFrame = SwipeEmbeddingDataFrame.__as__(single_letter_swipes(), verify)
print(f'generating data took {time() - starttime} seconds')

starttime = time()
convolved_data = data.convolve(fraction=len(data) - 1, verify=verify)
print(f'convolving took {time() - starttime} seconds')

preprocessor = Preprocessor()

metric = Metrics(preprocessor.preprocess(convolved_data), preprocessor.decode, convolved_data.get_i, len(data))

training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=20)  \
                                          .with_callback(metric)              \
                                          .fit(convolved_data)
