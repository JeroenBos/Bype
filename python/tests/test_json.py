import unittest
from keyboard._2_transform import Preprocessor
from keyboard._4a_word_input_model import CappedWordStrategy
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
                convolution_fraction=1.0,
                loss_ctor='binary_crossentropy',
                max_timesteps=None,
                word_input_strategy=CappedWordStrategy(n=5)
                )
            """.replace('\n', '').replace(' ', '').replace('\t', '').replace('\r', '')

        preprocessor = Preprocessor()
        representation = repr(preprocessor).replace(' ', '')
        assert representation == expected


    def test_preprocessor_decoder(self):
        preprocessor = json_decoders['Preprocessor']('"' + repr(Preprocessor()) + '"')
        assert isinstance(preprocessor, Preprocessor)


    def test_real_life_decode_error(self):
        json_contents = "\"Preprocessor(loss_ctor='binary_crossentropy',max_timesteps=2,word_input_strategy=CappedWordStrategy(n=5))\""

        preprocessor = json_decoders['Preprocessor'](json_contents)

        assert isinstance(preprocessor, Preprocessor)
        

if __name__ == '__main__':
    TestEncoder().test_real_life_decode_error()
