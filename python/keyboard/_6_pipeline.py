from python.keyboard._0_types import SwipeEmbeddingDataFrame
from python.keyboard._1a_generate import single_letter_swipes
from python.keyboard._2_transform import Preprocessor
from python.keyboard._3_model import KeyboardEstimator
from python.keyboard._4_scoring import Scorer, MyLoss
from python.keyboard._5_output import KeyboardResultWriter
from python.keyboard.hp import do_hp_search
from typing import List, Union


data = SwipeEmbeddingDataFrame.__as__(single_letter_swipes)

scorer = Scorer(data)
loss_fn = MyLoss(data)

preprocessor = Preprocessor(loss_ctor=loss_fn)

training = KeyboardEstimator[preprocessor].create_initialized().fit(data)


# ############# HYPER PARAMETERS #############
exit()

hp_space = KeyboardEstimator[None](
            num_epochs=[5, 6]
        ).params

result = do_hp_search(KeyboardEstimator[preprocessor],
                      data,
                      KeyboardResultWriter(),
                      hp_space,
                      scoring=scorer)
