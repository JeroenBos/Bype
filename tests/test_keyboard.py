import unittest
from python.keyboard.hp import do_hp_search, MyBaseEstimator, Models
from typing import List, Union
from python.keyboard.generic import generic
from python.model_training import InMemoryDataSource, ResultOutputWriter
import pandas as pd
import tensorflow as tf
from python.keyboard._0_types import T, Key, Keyboard, SwipeDataFrame
from python.keyboard._1_import import SPECs
from python.keyboard._2_transform import encode, decode
from python.keyboard._3_model import KeyboardEstimator
from tests.test_cluster import print_fully
import math


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
        estimator = KeyboardEstimator()
        estimator._create_model()

    def test_load_keyboard_layout(self):
        from python.keyboard._1_import import keyboard_layouts  # noqa
        assert len(keyboard_layouts) > 0

    def test_interpreting_keyboard_layout(self):
        from python.keyboard._1_import import KEYBOARD_LAYOUT_SPEC
        from python.keyboard._2_transform import get_keyboard

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
        from python.keyboard._1a_generate import generate_taps_for
        tap = generate_taps_for('a')
        assert isinstance(tap, pd.DataFrame)
        assert 'X' in tap.columns.values
        assert len(tap) == 1
        assert tap[SPECs.keyboard_layout][0] == 0
        assert tap[SPECs.x][0] == 54

    def test_swipe_embedding(self):
        from python.keyboard._1a_generate import create_empty_swipe_embedding_df
        df = create_empty_swipe_embedding_df(1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert len(df.columns) == 1
        assert df.columns.values[0] == 'swipes'

    def test_swipe_embedding_with_entries(self):
        from python.keyboard._1a_generate import create_empty_swipe_embedding_df, create_empty_swipe_df
        df = create_empty_swipe_embedding_df(1)
        df['swipes'][0] = create_empty_swipe_df(5)

        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 1
        assert df.columns.values[0] == 'swipes'
        assert len(df) == 1
        assert isinstance(df['swipes'][0], pd.DataFrame)
        entry = df['swipes'][0]
        assert len(entry) == 5

    def test_encode_single_letter(self):
        from python.keyboard._1a_generate import generate_taps_for, keyboards
        keyboard = keyboards[0]
        norm_x, norm_y = keyboard.normalize_x, keyboard.normalize_y

        swipe = generate_taps_for('a')
        features = encode(swipe, 'a')
        expected_features = [  # an item per touch event
                                [norm_x(54), norm_y(141),  # tap letter 'a'
                                 1,                       # length of the word 
                                 norm_x(54), norm_y(141),  # copy of the word
                                 -1, -1,                  # padding of the copy
                                 -1, -1]                  # padding of the copy
                            ]
        assert features == expected_features





class TDD(unittest.TestCase):
    def test_hp_search(self):

        class UglyEstimator(MyBaseEstimator):
            def __init__(self, num_epochs=5, activation='relu'):
                super().__init__()
                self.num_epochs = num_epochs
                self.activation = activation

            def _create_model(self) -> Models:
                return tf.keras.Sequential([
                    tf.keras.layers.Dense(14, activation=self.activation),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])

        ranges = UglyEstimator(num_epochs=[5, 6]).params

        df = pd.DataFrame(data=[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]], columns=['X', 'y'])
        do_hp_search(UglyEstimator,
                     InMemoryDataSource(df, 'y'),
                     ResultOutputWriter(),
                     ranges)

    def test_generating(self):
        from python.keyboard._1a_generate import single_letters_data  # noqa
        words, swipes = single_letters_data
        assert isinstance(words, pd.DataFrame)
        assert isinstance(swipes, pd.DataFrame)
        assert len(words.columns) == 1
        assert len(swipes.columns) > 10
        assert len(words) == len(swipes)
        assert swipes['X'][0] == 'a'
        assert math.isnan(swipes['Y'][0])


if __name__ == '__main__':
    Testkeyboard().test_encode_single_letter()
