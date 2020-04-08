import unittest
from python.keyboard.hp import do_hp_search, MyBaseEstimator, Model
from typing import List, Union
from python.keyboard.generic import generic
from python.model_training import InMemoryDataSource, ResultOutputWriter
import pandas as pd
import tensorflow as tf


class HpParamsTests(unittest.TestCase):
    def test_metaclass_indexer(self):
        ref_var = [0]

        class Meta(type):
            def __getitem__(self, key):
                ref_var[0] = 1
                return self

        class T(metaclass=Meta):
            pass

        Tprime = T[0]
        assert ref_var[0] == 1
        t = Tprime()
        assert isinstance(t, Tprime)
        assert isinstance(t, T)

    def test_metaclass_typeparameter(self):
        class Meta(type):

            # def __init__(cls, cls_name, cls_bases, cls_dict):
            #     super(Meta, cls).__init__(cls_name, cls_bases, cls_dict)

            def __getitem__(self, key):
                newcls = type(self.__name__, self.__bases__, dict(self.__dict__))
                newcls.T = key
                return newcls

        class T(metaclass=Meta):
            pass

        Tprime = T[0]
        assert Tprime.T == 0
        t = Tprime()
        assert t.T == 0
        Tdoubleprime = T[1]
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
        class T(metaclass=generic('TParams')):
            def get_param(cls):
                return cls.TParams

        assert T[int].TParams == int

    def test_access_to_type_param_on_self(self):
        class T(metaclass=generic('TParams')):
            def get_param(cls):
                return cls.TParams

        assert T[int]().TParams == int

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

    def test_hp_search(self):

        class UglyEstimator(MyBaseEstimator):
            def __init__(self, num_epochs=5, activation='relu'):
                super().__init__()
                self.num_epochs = num_epochs
                self.activation = activation

            def _create_model(self) -> Model:
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


if __name__ == '__main__':
    HpParamsTests().test_hp_search()
