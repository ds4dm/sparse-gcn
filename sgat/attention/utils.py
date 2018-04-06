# coding: utf-8

"""A collection of utilities."""


class Callable(type):
    """Callable Metaclass.

    A metaclass that ensure the class implements the __call__ function.
    """

    def __init__(self, name, bases, dict):
        """Initialization."""
        assert("__call__" in dict)
        super().__init__(name, bases, dict)
