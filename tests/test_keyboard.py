import unittest
from python.keyboard.hp import Params, MLModel, AbstractHpEstimator
from typing import List, Union
from python.keyboard.generic import generic


class TestMLModel(MLModel):
    def _create_model(self):
        return None


class HpParamsTests(unittest.TestCase):
    def test_params_clone(self):
        params = Params()
        paramsAsDict = params.getParameters()
        assert len(paramsAsDict) == 2
        assert 'activation' in paramsAsDict
        assert 'num_epochs' in paramsAsDict

    def test_params_subclass_clone(self):
        class TestParams(Params):
            def __init__(self,
                         num_epochs=5,
                         activation='relu',
                         test_param=''):
                super().__init__(num_epochs, activation)
                self.test_param = test_param

        params = TestParams()
        paramsAsDict = params.getParameters()
        assert len(paramsAsDict) == 3
        assert 'activation' in paramsAsDict
        assert 'num_epochs' in paramsAsDict
        assert 'test_param' in paramsAsDict

    def test_hpEstimator(self):
        estimator = AbstractHpEstimator(lambda params: TestMLModel(params))
        estimator.set_params({'activation': 'sigmoid'})
        estimator.set_params({'activation': 'relu'})

        assert len(estimator.models) == 2

    def test_hpEstimator_overriding_classmethod(self):
        class HpEstimator(AbstractHpEstimator):
            @classmethod
            def _get_param_names(cls):
                return sorted([])

        estimator = HpEstimator(lambda params: TestMLModel(params))
        result = estimator._get_param_names()
        assert len(result) == 0

    def test_hpEstimator_default_override__get_param_names(self):
        estimator = AbstractHpEstimator(lambda params: TestMLModel(params))
        result = estimator._get_param_names()
        assert len(result) == 2
        assert result[0] == 'activation'
        assert result[1] == 'num_epochs'

    # def test_hp_search(self):
    #     class HpEstimator(AbstractHpEstimator):
    #         @classmethod
    #         def _get_param_names(cls):
    #             return sorted(['only_param'])
    #     do_hp_search(HpEstimator(lambda params: TestMLModel(params), Params),
    #                  InMemoryDataSource(df, 0),
    #                  ResultOutputWriter(),
    #                  Params(

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

        Tprime = GenericType[Params]
        assert Tprime.U == Params
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


if __name__ == '__main__':
    HpParamsTests().test_generic_classmethod_overriding()
