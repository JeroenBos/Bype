import unittest
from MyBaseEstimator import MyBaseEstimator
from typing import List, Union
from generic import generic
from DataSource import InMemoryDataSource
import pandas as pd
import tensorflow as tf
from keyboard._0_types import Key, Keyboard, SwipeDataFrame, SwipeEmbeddingDataFrame, T
from keyboard._1_import import RawTouchEvent
from keyboard._2_transform import Preprocessor
from keyboard._3_model import KeyboardEstimator
from utilities import print_fully
import math


class IntOrHalfInt:
    def __init__(self, i: int):
        self.i = i

    def __eq__(self, value):
        if isinstance(value, int) or isinstance(value, float):
            return value == self.i \
                or value == self.i + 0.5 \
                or value == self.i - 0.5
        elif isinstance(value, self.__class__):
            return self.__eq__(self, value.i)
        else:
            return super().__eq__(value)

class TestKey(Key):
    def __init__(self, key: Key):
        def to_key(key):
            if key == 'edge_flags':
                return 'edgeFlags'
            return key
        super().__init__(**{to_key(key): value for key, value in key.__dict__.items()})

    @property
    def norm_x(self) -> IntOrHalfInt:
        """ Gets the horizontal middle of the button associated with the specified character """
        return self.keyboard.normalize_x(self.x + self.width / 2)

    @property
    def norm_y(self) -> IntOrHalfInt:
        """ Gets the vertical middle of the button associated with the specified character """
        return self.keyboard.normalize_y(self.y + self.height / 2)





class HpParamsTests(unittest.TestCase):
    def test_metaclass_indexer(self):
        ref_var = [0]

        class Meta(type):
            def __getitem__(self, key):
                ref_var[0] = 1
                return self

        class TestType(metaclass=Meta):
            pass

        Tprime = TestType[0]
        assert ref_var[0] == 1
        t = Tprime()
        assert isinstance(t, Tprime)
        assert isinstance(t, TestType)

    def test_metaclass_typeparameter(self):
        class Meta(type):

            # def __init__(cls, cls_name, cls_bases, cls_dict):
            #     super(Meta, cls).__init__(cls_name, cls_bases, cls_dict)

            def __getitem__(self, key):
                newcls = type(self.__name__, self.__bases__, dict(self.__dict__))
                newcls.T = key
                return newcls

        class TestType(metaclass=Meta):
            pass

        Tprime = TestType[0]
        assert Tprime.T == 0
        t = Tprime()
        assert t.T == 0
        Tdoubleprime = TestType[1]
        assert Tdoubleprime.T == 1
        u = Tdoubleprime()
        assert u.T == 1
        assert t.T == 0

    def test_metaclass_typeparameter_by_name(self):
        def getMeta(params: Union[str, List[str]] = ['T']):
            if len(params) == 0:
                raise ValueError('At least one type parameter name must be specified')
            if isinstance(params, str):
                params = [params]

            class Meta(type):
                def __getitem__(self, *args):
                    if(len(args) != len(params)):
                        raise ValueError(f"{len(params)} type arguments must be specified.")
                    for arg in args:
                        if not isinstance(arg, type):
                            raise ValueError()

                    name = ", ".join(params)
                    newcls = type(self.__name__ + '<' + name + '>', self.__bases__, dict(self.__dict__))
                    for typeArg, name in zip(args, params):
                        setattr(newcls, name, typeArg)
                    return newcls
            return Meta

        class GenericType(metaclass=getMeta('U')):
            pass

        Tprime = GenericType[int]
        assert Tprime.U == int
        Tdoubleprime = GenericType[type]
        assert Tdoubleprime.U == type

    def test_access_to_type_param_on_cls(self):
        class U(metaclass=generic('TParams')):
            def get_param(self):
                return self.TParams

        assert U[int].TParams == int

    def test_access_to_type_param_on_self(self):
        class U(metaclass=generic('TParams')):
            def get_param(self):
                return self.TParams

        assert U[int]().TParams == int

    def test_generic_classmethod_overriding(self):
        class GenericBaseType(metaclass=generic()):
            @classmethod
            def _get_param_names(cls):
                return cls.__name__

        class GenericType(GenericBaseType):
            @classmethod
            def _get_param_names(cls):
                return cls.__name__

        a = GenericBaseType[int]()._get_param_names()
        b = GenericType[int]()._get_param_names()
        assert a == 'GenericBaseType<T>'
        assert b == 'GenericType<T>'

    def test_generic_classmethod_overriding_with_different_cls(self):
        class GenericBaseType(metaclass=generic()):
            @classmethod
            def _get_param_names(cls):
                return cls.__name__

        class GenericType(GenericBaseType):
            @classmethod
            def _get_param_names(cls):
                return super(cls, cls)._get_param_names.__func__(cls.T)

        a = GenericBaseType[int]()._get_param_names()
        b = GenericType[int]()._get_param_names()
        assert a == 'GenericBaseType<T>'
        assert b == 'int'

    def test_get_id(self):
        class UglyEstimator(MyBaseEstimator):
            def __init__(self, num_epochs=5, activation='relu'):
                super().__init__()
                self.num_epochs = num_epochs
                self.activation = activation

        repr = UglyEstimator()._get_params_repr()
        assert repr == "(activation='relu', num_epochs=5)"


class Testkeyboard(unittest.TestCase):
    def test_can_create_model(self):
        estimator = KeyboardEstimator[Preprocessor()].create_initialized()
        estimator._create_model()

    def test_load_keyboard_layout(self):
        from keyboard._1_import import keyboard_layouts
        assert len(keyboard_layouts) > 0

    def test_interpreting_keyboard_layout(self):
        from keyboard._1_import import KEYBOARD_LAYOUT_SPEC
        from keyboard._2_transform import get_keyboard

        test_data = {
            "codes": [1, 3],
            "x": 1,
            "y": 2,
            "width": 10,
            "height": 10,
            "edgeFlags": 0,
            "repeatable": True,
            "toggleable": True,
        }

        df = pd.DataFrame([[None for _ in test_data]], columns=list(KEYBOARD_LAYOUT_SPEC.keys()))
        for key, value in test_data.items():
            df[key][0] = value
        assert len(df.columns) == len(test_data)
        assert len(df) == 1
        assert df['x'][0] == 1
        assert isinstance(df['codes'][0], List)

        keyboard = get_keyboard(df)
        assert len(keyboard) == 2
        assert 0 not in keyboard
        assert 1 in keyboard
        assert 2 not in keyboard
        assert 3 in keyboard
        assert isinstance(keyboard[1], Key)
        assert isinstance(keyboard[3], Key)
        assert keyboard[1].code == 1
        assert keyboard[1].y == 2
        assert keyboard[3].y == 2

    def test_generate_single_letters(self):
        from keyboard._1a_generate import generate_taps_for
        tap = generate_taps_for('a')
        assert isinstance(tap, pd.DataFrame)
        assert 'X' in tap.columns.values
        assert len(tap) == 1
        assert tap.KeyboardLayout[0] == 0
        assert tap.X[0] == 108

    def test_swipe_embedding(self):
        df = SwipeEmbeddingDataFrame.create_empty(1)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df, SwipeEmbeddingDataFrame)
        assert len(df) == 1
        assert sorted(df.columns.values) == sorted(['swipes', 'words', 'correct'])

    def test_swipe_embedding_with_entries(self):
        df = SwipeEmbeddingDataFrame.create_empty(1)
        df.swipes[0] = SwipeDataFrame.create_empty(5)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert isinstance(df.swipes[0], pd.DataFrame)
        entry = df.swipes[0]
        assert len(entry) == 5

    def test_encode_single_letter(self):
        from keyboard._1a_generate import generate_taps_for, keyboards

        a = TestKey(keyboards[0].get_key('a'))
        swipe = generate_taps_for('a')
        features = Preprocessor().encode(swipe, 'a')
        expected_features = [  # an item per touch event
                                [a.norm_x, a.norm_y,  # tap letter 'a'
                                 0.2,                 # relative word length 
                                 a.norm_x, a.norm_y,  # copy of the word
                                 -1, -1,              # padding of the word
                                 -1, -1,              # padding of the word
                                 -1, -1,              # padding of the word
                                 -1, -1]              # padding of the word
                            ]
        assert features == expected_features

    def test_encode_double_letters(self):
        from keyboard._1a_generate import generate_taps_for, keyboards
        # arrange
        a = TestKey(keyboards[0].get_key('a'))
        b = TestKey(keyboards[0].get_key('b'))
        expected_features = [
            [  # touch event for 'a'
                a.norm_x, a.norm_y,    # tap letter 'a'
                0.4,                   # relative word length 
                a.norm_x, a.norm_y,    # copy of the word
                b.norm_x, b.norm_y,    # copy of the word
                -1, -1,                # padding of the word
                -1, -1,                # padding of the word
                -1, -1],               # padding of the word
            [  # touch event for 'b'
                b.norm_x, b.norm_y,    # tap letter 'b'
                0.4,                   # relative word length 
                a.norm_x, a.norm_y,    # copy of the word
                b.norm_x, b.norm_y,    # copy of the word
                -1, -1,                # padding of the word
                -1, -1,                # padding of the word
                -1, -1,                # padding of the word
            ]
        ]

        # act
        swipe = generate_taps_for('ab')
        assert len(swipe) == 2
        features = Preprocessor(max_timesteps=2).encode(swipe, 'ab')

        # assert
        diffs = [f"({i}, {j}): {features[i][j]} whereas expected: {expected_features[i][j]}"  # noqa
                 for i in range(len(expected_features)) 
                 for j in range(len(expected_features[i]))
                 if features[i][j] != expected_features[i][j]]
        assert features == expected_features


if __name__ == '__main__':
    Testkeyboard().test_encode_double_letters()
