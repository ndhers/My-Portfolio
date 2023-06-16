"""
Functions that can be used on symbols
"""
from typing import List
import numpy as np

from .sym import Symbol, symbol, symbolic

NumberTypes = (int, float, complex)

# concatenation operator for vector functions


def sum(v: 'List[Symbol]') -> Symbol:
    """
   Summing elements of a list of Symbols

   :param v: list of Symbol objects
   :type v: list of Symbols
   :return: summation of elements
   :rtype: Symbol
   """
    if len(v) == 1:
        return v[0]
    result = v[0]
    for num in v[1:]:
        result = result + num
    return result


def prod(v: 'List[Symbol]') -> Symbol:
    """
   Multiplying elements of a list of Symbols

   :param v: list of Symbol objects
   :type v: list of Symbols
   :return: multiplicatio of elements
   :rtype: Symbol
   """
    if len(v) == 1:
        return v[0]
    if len(v) == 2:
        return v[0] * v[1]
    else:
        return v[0] * prod(v[1:])


# For one or two input functions
# trigonometric functions

@symbolic
def sin(a: float, deriv=False) -> float:
    """
   Sine operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after sine operation
   :rtype: Symbol
   """
    if deriv:
        return np.cos(a)
    return np.sin(a)


@symbolic
def cos(a: float, deriv=False) -> float:
    """
   Cosine operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after cosine operation
   :rtype: Symbol
   """
    if deriv:
        return -np.sin(a)
    return np.cos(a)


@symbolic
def tan(a: float, deriv=False) -> float:
    """
   Tangent operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after tangent operation
   :rtype: Symbol
   """
    if deriv:
        return 1 / (np.cos(a)) ** 2
    return np.tan(a)


@symbolic
def arcsin(a: float, deriv=False) -> float:
    """
   Arcsine operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after arcsine operation
   :rtype: Symbol
   """
    if deriv:
        return 1 / (1 - a ** 2) ** 0.5
    return np.arcsin(a)


@symbolic
def arccos(a: float, deriv=False) -> float:
    """
   Arccosine operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after arccosine operation
   :rtype: Symbol
   """
    if deriv:
        return -1 / (1 - a ** 2) ** 0.5
    return np.arccos(a.real)


@symbolic
def arctan(a: float, deriv=False) -> float:
    """
   Arctangent operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after arctangent operation
   :rtype: Symbol
   """
    if deriv:
        return 1 / (1 + a ** 2)
    return np.arctan(a)


# Hyperbolic functions

@symbolic
def sinh(a: float, deriv=False) -> float:
    """
   Sinh operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after sinh operation
   :rtype: Symbol
   """
    if deriv:
        return np.cosh(a)
    return np.sinh(a)


@symbolic
def cosh(a: float, deriv=False) -> float:
    """
   Cosh operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after cosh operation
   :rtype: Symbol
   """
    if deriv:
        return np.sinh(a)
    return np.cosh(a)


@symbolic
def tanh(a: float, deriv=False) -> float:
    """
   Tanh operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after tanh operation
   :rtype: Symbol
   """
    if deriv:
        return 1 / (np.cosh(a)) ** 2
    return np.tanh(a)


@symbolic
def sqrt(a: float, deriv=False) -> float:
    """
   Square root operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after square root operation
   :rtype: Symbol
   """
    if deriv:
        return 1 / (2 * np.sqrt(a))
    return np.sqrt(a)


# Natural functions

@symbolic
def exp(a: float, deriv=False) -> float:
    """
   Exponential operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after exponential operation
   :rtype: Symbol
   """
    return np.exp(a)


@symbolic
def log(a: float, deriv=False) -> float:
    """
   Natural logarithmic operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after natural logarithmic operation
   :rtype: Symbol
   """
    if deriv:
        return 1 / a
    return np.log(a)


@symbolic
def log10(a: float, deriv=False) -> float:
    """
   Base 10 logarithmic operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after base 10 logarithmic operation
   :rtype: Symbol
   """
    if deriv:
        return 1 / (a * np.log(10))
    return np.log10(a)


# define a log function with user defined base
def log_base(a: Symbol, base) -> Symbol:
    """
   Logarithm base 'base' operation on Symbol object, returning
   either function or derivative value.

   :param a: input Symbol
   :type a: Symbol
   :param float base: base floating point number, or a symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after logarithm base 'base' operation
   :rtype: Symbol
   """

    if isinstance(base, Symbol):
        return log(a)/log(base)
    elif isinstance(base, NumberTypes):
        if base <= 0:
            raise ValueError('Base cannot be lower than 0')
        else:
            return log(a) / np.log(base)


# Activation fuctions

@symbolic
def sigmoid(a: float, deriv=False) -> float:
    """
   Sigmoid operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after sigmoid operation
   :rtype: Symbol
   """
    val = 1 / (1 + np.exp(- a))
    if deriv:
        return val * (1 - val)
    return val


@symbolic
def ReLU(a: float, deriv=False) -> float:
    """
   ReLU operation on Symbol object, returning
   either function or derivative value.

   :param a: input float number
   :type a: Symbol
   :param deriv: returning derivative, boolean type
   :type deriv: bool
   :return: function or derivative value after ReLU operation
   :rtype: Symbol
   """
    if a > 0:
        if deriv:
            return 1.
        return a
    else:
        return 0.
