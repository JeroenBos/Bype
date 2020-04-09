# this file generates training data

from python.keyboard._1_import import SPEC
import pandas
from pandas import DataFrame


single_letters = [chr(i) for i in range(97, 97 + 26)]
single_letter_df: DataFrame = pandas.read_csv('/home/jeroen/git/bype/data/empty.csv',
                                              names=SPEC.keys(),
                                              dtype=SPEC)
