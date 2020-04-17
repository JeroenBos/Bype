import unittest
from keyboard._3a_word_input_model import CappedWordStrategy
from myjson import decoders

class TestEncoder(unittest.TestCase):
    def test_encode_CappedWord(self):
        obj = CappedWordStrategy(n=10)
        assert obj.__repr__() == 'CappedWordStrategy(n=10)'

    def test_decode_CappedWord(self):
        s = '"CappedWordStrategy(n=10)"'

        result = decoders['CappedWordStrategy'](s)

        assert isinstance(result, CappedWordStrategy)
        assert result.n == 10


    def test_decode_CappedWord_from_fields(self):
        s = '{ "n": 10 }'

        result = decoders['CappedWordStrategy'](s)

        assert isinstance(result, CappedWordStrategy)
        assert result.n == 10


if __name__ == '__main__':
    TestEncoder().test_decode_CappedWord_from_fields()
