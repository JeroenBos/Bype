# this file generates training data

from python.keyboard._1_import import SPEC
import pandas as pd
from pandas import DataFrame
from python.model_training import InMemoryDataSource, TrivialDataSource  # noqa


swipes_embedding_df: DataFrame = pd.read_csv('/home/jeroen/git/bype/data/empty.csv',
                                             names=SPEC.keys(),
                                             dtype=SPEC)

df = DataFrame(None, columns=['words', 'swipes'], dtype=object)


single_letters = [chr(i) for i in range(97, 97 + 26)]
single_letter_words = DataFrame(single_letters, columns=['words'], dtype=str)
single_letter_swipes = swipes_embedding_df.append(DataFrame(single_letters, columns=['X']))

single_letters_data = TrivialDataSource(single_letter_words, single_letter_swipes)
