from python.keyboard._2_transform import data
from python.keyboard._3_model import KeyboardEstimator
from python.keyboard._4_scoring import score_function
from python.keyboard._5_output import KeyboardResultWriter
from python.keyboard.hp import do_hp_search
from typing import List, Union  # noqa

# estimator = KeyboardEstimator()
# result = estimator.fit_data_source(data)
# score_function(estimator)

# print(result)
# print('DONE')

# exit()

ranges = KeyboardEstimator(
            num_epochs=[5, 6]
        )
result = do_hp_search(KeyboardEstimator,
                      data,
                      KeyboardResultWriter(),
                      ranges.params,
                      scoring=score_function)
