from keyboard._0_types import Input
from keyboard._1_import import keyboards, raw_data
from utilities import concat, is_list_of, read_all, split_at, split_by
from typing import List

def text_file_to_swipes(path: str) -> List[str]:
    raw_data_text = read_all(path)
    return text_to_swipes(raw_data_text)

def text_to_swipes(raw_data_text: str) -> List[str]:

    def split_punctuation(s: str):
        assert isinstance(s, str)
        s = s.replace(' ', '').replace('\r', '')
        split_indices = concat((i, i + 1) for i, c in enumerate(s) if not c.isalpha())

        return split_at(s, *split_indices)

    def is_valid(s: str):
        return len(s) != 0 and not s.isspace()

    words = split_by(raw_data_text, '\n', ' ')  # word here means more generic than string of characters: it's more of a string of values (values represented on keys)
    words = concat(split_punctuation(word) for word in words)
    words = [word for word in words if is_valid(word)]
    return words


print(text_file_to_swipes('/home/jeroen/git/bype/data/2020-03-20_0.txt'))
