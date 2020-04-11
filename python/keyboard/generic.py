def generic(*params: str) -> type:
    """
    Creates a metatype representing a class with the specified type parameters.
    Example usages:

    class GenericType(metaclass=generic('T', 'U')):
        pass

    class GenericTypeWithSuperclass(metaclass=generic('T', 'U'), superclass):
        pass

    also note that the superclass can be accessed via super(self.__class__, self) and super(cls, cls)
    """
    if len(params) == 0:
        params = ['T']
    if isinstance(params, str):
        params = [params]

    class Metatype(type):
        def __getitem__(self, *args):
            if(len(args) != len(params)):
                raise ValueError(f"{len(params)} type arguments must be specified.")
            # for arg in args:
            #     if not isinstance(arg, type):
            #         raise ValueError()

            newcls = type(f"{self.__name__}<{', '.join(params)}>", self.__bases__, dict(self.__dict__))
            for typeArg, name in zip(args, params):
                setattr(newcls, name, typeArg)
            return newcls
    return Metatype
