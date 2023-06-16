"""
Functions that can be used on symbols
"""
import numpy as np

from .sym import DualNumber, Symbol, symbol, symbolic

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
   :return: multiplication of elements
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
def sin(a: DualNumber) -> DualNumber:
    """
   Sine operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after sine operation
   :rtype: Symbol
   """
    real = np.sin(a.real)
    dual = np.cos(a.real) * a.dual
    return DualNumber(real, dual)


@symbolic
def cos(a: DualNumber) -> DualNumber:
    """
   Cosine operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after cosine operation
   :rtype: Symbol
   """
    real = np.cos(a.real)
    dual = -np.sin(a.real) * a.dual
    return DualNumber(real, dual)


@symbolic
def tan(a: DualNumber) -> DualNumber:
    """
   Tangent operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after tangent operation
   :rtype: Symbol
   """
    real = np.tan(a.real)
    dual = a.dual / (np.cos(a.real)) ** 2
    return DualNumber(real, dual)


@symbolic
def arcsin(a: DualNumber) -> DualNumber:
    """
   Arcsine operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after arcsine operation
   :rtype: Symbol
   """
    real = np.arcsin(a.real)
    dual = a.dual / (1 - a.real ** 2) ** 0.5
    return DualNumber(real, dual)


@symbolic
def arccos(a: DualNumber) -> DualNumber:
    """
   Arccosine operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after arccosine operation
   :rtype: Symbol
   """
    real = np.arccos(a.real)
    dual = -a.dual / (1 - a.real ** 2) ** 0.5
    return DualNumber(real, dual)


@symbolic
def arctan(a: DualNumber) -> DualNumber:
    """
   Arctangent operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after arctangent operation
   :rtype: Symbol
   """
    real = np.arctan(a.real)
    dual = a.dual / (1 + a.real ** 2)
    return DualNumber(real, dual)


# Hyperbolic functions
@symbolic
def sinh(a: DualNumber) -> DualNumber:
    """
   Sinh operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after sinh operation
   :rtype: Symbol
   """
    real = np.sinh(a.real)
    dual = a.dual * np.cosh(a.real)
    return DualNumber(real, dual)


@symbolic
def cosh(a: DualNumber) -> DualNumber:
    """
   Cosh operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after cosh operation
   :rtype: Symbol
   """
    real = np.cosh(a.real)
    dual = a.dual * np.sinh(a.real)
    return DualNumber(real, dual)


@symbolic
def tanh(a: DualNumber) -> DualNumber:
    """
   Tanh operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after tanh operation
   :rtype: Symbol
   """
    real = np.tanh(a.real)
    dual = a.dual * (1 - real ** 2)
    return DualNumber(real, dual)


# Natural functions

@symbolic
def sqrt(a: DualNumber) -> DualNumber:
    """
   Square root operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after square root operation
   :rtype: Symbol
   """
    real = np.sqrt(a.real)
    dual = a.dual / (2 * np.sqrt(a.real))
    return DualNumber(real, dual)


@symbolic
def exp(a: DualNumber) -> DualNumber:
    """
   Exponential operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after exponential operation
   :rtype: Symbol
   """
    real = np.exp(a.real)
    dual = real * a.dual
    return DualNumber(real, dual)


@symbolic
def log(a: DualNumber) -> DualNumber:
    """
   Natural logarithm operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after natural logarithm operation
   :rtype: Symbol
   """
    real = np.log(a.real)
    dual = a.dual / a.real
    return DualNumber(real, dual)


@symbolic
def log10(a: DualNumber) -> DualNumber:
    """
   Logarithm base 10 operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after logarithm base 10 operation
   :rtype: Symbol
   """
    real = np.log10(a.real)
    dual = a.dual / (np.log(10) * a.real)
    return DualNumber(real, dual)


# define a log function with user defined base
def log_base(a: Symbol, base):
    """
   Logarithm base 'base' operation on Symbol object
   
   :param float base: base number of type float
   :param a: input dual number
   :type a: Symbol
   :return: dual number after logarithm base 'base' operation
   :rtype: Symbol
   """
    if isinstance(base, Symbol):
        return log(a) / log(base)
    elif isinstance(base, NumberTypes):
        if base <= 0:
            raise ValueError('Base cannot be lower than 0')
        return log(a) / np.log(base)
    else:
        return NotImplemented


# Activation functions

@symbolic
def sigmoid(a: DualNumber) -> DualNumber:
    """
   Sigmoid operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after sigmoid operation
   :rtype: Symbol
   """
    real = 1 / (1 + np.exp(- a.real))
    dual = a.dual * real * (1 - real)
    return DualNumber(real, dual)


@symbolic
def ReLU(a: DualNumber) -> DualNumber:
    """
   ReLU operation on Symbol object

   :param a: input dual number
   :type a: Symbol
   :return: dual number after ReLU operation
   :rtype: Symbol
   """
    if a.real > 0:
        # create a copy of a
        return DualNumber(a.real, a.dual)
    if a.real <= 0:
        return DualNumber(0, 0)
