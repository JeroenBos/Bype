import unittest
from keyboard._2_transform import Preprocessor
from keyboard._3a_word_input_model import CappedWordStrategy
from myjson import json_decoders

class TestEncoder(unittest.TestCase):
    def test_encode_CappedWord(self):
        obj = CappedWordStrategy(n=10)
        assert obj.__repr__() == 'CappedWordStrategy(n=10)'

    def test_decode_CappedWord(self):
        s = '"CappedWordStrategy(n=10)"'

        result = json_decoders['CappedWordStrategy'](s)

        assert isinstance(result, CappedWordStrategy)
        assert result.n == 10


    def test_decode_CappedWord_from_fields(self):
        s = '{ "n": 10 }'

        result = json_decoders['CappedWordStrategy'](s)

        assert isinstance(result, CappedWordStrategy)
        assert result.n == 10

    def test_preprocessor_repr(self):
        expected = """Preprocessor(
                batch_count=1,
                loss_ctor='binary_crossentropy',
                max_timesteps=1,
                swipe_feature_count=13,
                word_input_strategy=CappedWordStrategy(n=5)
                )
            """.replace('\n', '').replace(' ', '').replace('\t', '').replace('\r', '')

        preprocessor = Preprocessor()
        representation = repr(preprocessor).replace(' ', '')
        assert representation == expected



if __name__ == '__main__':
    TestEncoder().test_preprocessor_repr()
