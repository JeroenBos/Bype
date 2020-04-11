from python.keyboard._0_types import SwipeEmbeddingDataFrame
from python.keyboard._1a_generate import single_letter_swipes
from python.keyboard._2_transform import Preprocessor
from python.keyboard._3_model import KeyboardEstimator
from python.keyboard._4_scoring import Scorer
from python.keyboard._5_output import KeyboardResultWriter
from python.keyboard.hp import do_hp_search
from typing import List, Union


data = SwipeEmbeddingDataFrame.__as__(single_letter_swipes)

hp_space = KeyboardEstimator(
            num_epochs=[5, 6]
        ).params

scorer = Scorer(data)
preprocessor = Preprocessor()


result = do_hp_search(lambda **initial_params: KeyboardEstimator.create(preprocessor, **initial_params),
                      data,
                      KeyboardResultWriter(),
                      hp_space,
                      scoring=scorer)
