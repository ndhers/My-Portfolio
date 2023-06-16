"""
Functions that can be used on symbols
"""
import numpy as np

from ..forward.sym import Symbol
from .sym import SymVec, symbolic_vec
from ..forward import function as fw_f


# trigonometric functions

@symbolic_vec
def sin(a: Symbol) -> Symbol:
    """
   Sine operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after Sine operation
   :rtype: Symbol
   """
    return fw_f.sin(a)


@symbolic_vec
def cos(a: Symbol) -> Symbol:
    """
   CoSine operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after CoSine operation
   :rtype: Symbol
   """
    return fw_f.cos(a)


@symbolic_vec
def tan(a: Symbol) -> Symbol:
    """
   Tangent operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after tangent operation
   :rtype: Symbol
   """
    return fw_f.tan(a)


@symbolic_vec
def arcsin(a: Symbol) -> Symbol:
    """
   ArcSine operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after arcsine operation
   :rtype: Symbol
   """
    return fw_f.arcsin(a)


@symbolic_vec
def arccos(a: Symbol) -> Symbol:
    """
   ArcCosine operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after arcCosine operation
   :rtype: Symbol
   """
    return fw_f.arccos(a)


@symbolic_vec
def arctan(a: Symbol) -> Symbol:
    """
   ArcTangent operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after arctangent operation
   :rtype: Symbol
   """
    return fw_f.arctan(a)


# Hyperbolic functions
@symbolic_vec
def sinh(a: Symbol) -> Symbol:
    """
   Hyperbolic Sine operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after Hyperbolic Sine operation
   :rtype: Symbol
   """
    return fw_f.sinh(a)


@symbolic_vec
def cosh(a: Symbol) -> Symbol:
    """
   Hyperbolic CoSine operation on SymVec object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after Hyperbolic coSine operation
   :rtype: Symbol
   """
    return fw_f.cosh(a)


@symbolic_vec
def tanh(a: Symbol) -> Symbol:
    """
   Tanh operation on Symbol object

   :param a: input Symbol
   :type a: Symbol
   :return: ® after tanh operation
   :rtype: Symbol
   """
    return fw_f.tanh(a)


# Natural functions

@symbolic_vec
def sqrt(a: Symbol) -> Symbol:
    """
   Square root operation on Symbol object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after square root operation
   :rtype: Symbol
   """
    return fw_f.sqrt(a)


@symbolic_vec
def exp(a: Symbol) -> Symbol:
    """
   Exponential operation on Symbol object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after exponential operation
   :rtype: Symbol
   """
    return fw_f.exp(a)


@symbolic_vec
def log(a: Symbol) -> Symbol:
    """
   Natural logarithm operation on Symbol object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after natural logarithm operation
   :rtype: Symbol
   """
    return fw_f.log(a)


@symbolic_vec
def log10(a: Symbol) -> Symbol:
    """
   Logarithm base 10 operation on Symbol object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after logarithm base 10 operation
   :rtype: Symbol
   """
    return fw_f.log10(a)


# define a log function with user defined base
@symbolic_vec
def log_base(a: Symbol, base) -> Symbol:
    """
   Logarithm base 'base' operation on Symbol object

   :param float base: base number of type float
   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after logarithm base 'base' operation
   :rtype: Symbol
   """
    return fw_f.log_base(a, base)


# Activation functions

@symbolic_vec
def sigmoid(a: Symbol) -> Symbol:
    """
   Sigmoid operation on Symbol object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after sigmoid operation
   :rtype: Symbol
   """
    return fw_f.sigmoid(a)


@symbolic_vec
def ReLU(a: Symbol) -> Symbol:
    """
   ReLU operation on Symbol object

   :param a: input Symbol
   :type a: Symbol
   :return: Symbol after ReLU operation
   :rtype: Symbol
   """
    return fw_f.ReLU(a)
