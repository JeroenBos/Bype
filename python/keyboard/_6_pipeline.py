import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import single_letter_swipes, double_letter_swipes, single_and_double_letter_swipes, single_double_and_triple_letter_swipes  # noqa
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics
from keyboard._4_model import KeyboardEstimator
from typing import List, Union
from time import time
from tensorflow.keras.callbacks import LearningRateScheduler  # noqa

verify = False

starttime = time()
data: SwipeEmbeddingDataFrame = SwipeEmbeddingDataFrame.__as__(single_letter_swipes(), verify)
print(f'generating data took {time() - starttime} seconds')

starttime = time()
convolved_data = data.convolve(fraction=len(data) - 1, verify=verify)
print(f'convolving took {time() - starttime} seconds')

preprocessor = Preprocessor(max_timesteps=1)

metric = Metrics(preprocessor.preprocess(convolved_data), preprocessor.decode, convolved_data.get_i, len(data))
def adapt_learning_rate(epoch: int) -> float:
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.001
    elif epoch < 80:
        return 0.0001
    else:
        return 0.00001


lr_scheduler = LearningRateScheduler(adapt_learning_rate)


training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=100)  \
                                          .with_callback(metric, lr_scheduler)              \
                                          .fit(convolved_data)
