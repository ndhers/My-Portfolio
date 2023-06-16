"""
This module implements vector functions for automatic differentiation, building
on the Symbols from the `autodiff.forward` module.
"""

from typing import List, Union, Callable
import numpy as np
import itertools
from ..forward.sym import Symbol, symbol
NumberTypes = Union[int, float, complex]


def symbolic_vec(op: Callable[[Symbol, Union[None, NumberTypes, Symbol]], Symbol]) -> Callable[[Union['SymVec', Symbol], Union[Symbol, None]], Union['SymVec', Symbol]]:
    """
    Decorator that considers various type checking and
    converts a function of symbol or number to function for symbol vector.

    :param op: function that takes input of symbol and symbol or number, outputing symbol
    :type op: Callable[['SymVec', Union['Symbol', None]], np.ndarray]
    :return: wrapped function
    :rtype: Callable[[Union['SymVec', Symbol], Union[Symbol, None]], Union['SymVec', Symbol]]
    """

    def sym_vec_f(left: Union['SymVec', Symbol], right=None) -> Union['SymVec', Symbol]:
        # single element operation, apply op to each symbol within SymVec
        if right is None:
            if isinstance(left, SymVec):
                return SymVec([op(left.symbols[i]) for i in range(left.shape)],
                              names=left.names)
            else:
                return op(left)
        # double element operation, apply op based on different cases
        if isinstance(right, Symbol):
            return SymVec([op(s, right) for s in left.symbols], names=left.names + list(right.names))
        elif isinstance(right, SymVec):
            if right.shape != left.shape:
                raise IndexError('Dimension incorrect')
            else:
                return SymVec([op(left.symbols[i], right.symbols[i]) for i in range(left.shape)],
                              names=left.names + right.names)
        elif isinstance(right, np.ndarray):
            if right.ndim == 1 and right.shape[0] == left.shape:
                return SymVec([op(left.symbols[i], right[i]) for i in range(left.shape)],
                              names=left.names)
            else:
                raise IndexError('Dimension incorrect')
        else:
            # if right is a float or int
            return SymVec([op(s, right) for s in left.symbols], names=left.names)

    return sym_vec_f


@symbolic_vec
def _add(left, right):
    return left + right


@symbolic_vec
def _multiply(left, right):
    return left * right


@symbolic_vec
def _neg(left):
    return -left


@symbolic_vec
def _div(left, right):
    return left / right


@symbolic_vec
def _rdiv(left, right):
    return right / left


@symbolic_vec
def _pow(left, right):
    return left ** right


@symbolic_vec
def _rpow(left, right):
    return right ** left


def concat(symbols: List[Symbol]) -> 'SymVec':
    """
   concatenating a list of Symbols into a vector
   :param symbols: list of Symbol objects
   :type: list of Symbols
   :return: Symbol vector
   :rtype: SymVec
   """
    return SymVec(symbols)


class vec_gen:
    def __init__(self):
        self.counter = itertools.count()

    def generate(self, dim: int) -> 'SymVec':
        """
        Generate dummy Symbol vector with automatic name assignment
        can generate up to 99999 symbols in total.
        :param dim
        :type: int
        :return: Symbol vector
        :rtype: SymVec
        """
        if dim < 1:
            raise ValueError('dimension must be at least 1')
        result = SymVec(
            [Symbol(name='dum_' + str(next(self.counter)).zfill(5)) for i in range(dim)])
        return result


def getnames(s: Union[Symbol, 'SymVec']) -> List[str]:
    """Helper method to extract the names from a Symbol or SymVec object

    :param s: entity from which to extract the names
    :type s: Union[Symbol, 'SymVec]
    :return: list of names
    :rtype: List[str]
    """
    if s.name is None:
        return list(s.names)
    return [s.name]


class SymVec:
    # This placed our class at a higher priority than the np array and therefore our radd and rmul will be used
    __array_priority__ = 1

    def __init__(self, symbols: List[Symbol], names=None):
        """
        Initiate a symbol vector by passing in a list of symbols and the names
        of all symbols relevant. Note, only advanced users with very
        specific applications in mind should construct a SymVec object directly.
        In the vast majority of cases, Symbols should be constructed through the
        factory method `vec_gen()` or `concat()`.

        :param symbols: a list of Symbols
        :type: List[Symbol]
        :param names: the names of all relevant symbol, to keep track for quickeval and quick eriv
        """
        self.symbols = symbols
        if names is None:
            all_name = []
            for s in self.symbols:  # get the names of all relevant symbols if names is not passed in
                all_name = all_name + getnames(s)
            self.names = sorted(list(set(all_name)))
        else:
            self.names = sorted(list(set(names)))
        self.in_shape = len(self.names)
        self.shape = len(symbols)

    def eval(self, val: dict) -> np.array:
        """
        Evaluating the value of the symbol vector based on input values
        :param val: dictionary of the value for all the symbols
        :type: dict of {symbol_name : value}
        :return: each individual symbol evaluation result
        :rtype: np.array
        """
        return np.array([s.eval(val) for s in self.symbols])

    def quickeval(self, array_: np.ndarray) -> 'SymVec':
        """
        a quick evaluation of the symbol vector, map the input array to a sorted list of symbols
        :param array_: an 1-D np array which represent the values of individual parameters
        :type: np.ndarray
        :return: each individual symbol evaluation result
        :rtype: SymVec
        """

        if array_.ndim != 1:  # we only support vector for now
            raise TypeError('incorrect dimension')
        val_v = dict(zip(self.names, array_))
        return np.array([s.eval(val_v) for s in self.symbols])

    def deriv(self, val: dict):
        """
        Calculate the derivative of a symbol vector
        :param val: list of Symbol objects
        :type: list of Symbols
        :return: Jacobian Matrix
        :rtype: np.ndarray
        """
        return np.array([list(s.deriv(val).values()) for s in self.symbols])

    def quickderiv(self, val: dict):
        """
        Calculate the derivative of a symbol vector
        :param val: an 1-D np array which represent the values of individual parameters
        :type: list of Symbols
        :return: Jacobian Matrix
        :rtype: np.ndarray
        """
        val_v = dict(zip(self.names, val))
        return np.array([list(s.deriv(val_v).values()) for s in self.symbols])

    def sum(self):
        """
        sum the elements within a SymVec
        :param self
        :type: self
        :return: sum of Symbols
        :rtype: Symbol
        """
        result = 0
        for s in self.symbols:
            result = result + s
        return result

    def prod(self):
        """
        calculate the product of elements within a SymVec
        :param self
        :type: self
        :return: product of Symbols
        :rtype: Symbol
        """
        result = 1
        for s in self.symbols:
            result = result * s
        return result

    def __add__(self, other):
        return _add(self, other)

    def __sub__(self, other):
        return _add(self, -other)

    def __mul__(self, other):
        return _multiply(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return _add(-self, other)

    def __neg__(self):
        return _neg(self)

    def __truediv__(self, other):
        return _div(self, other)

    def __rtruediv__(self, other):
        return _rdiv(self, other)

    def __pow__(self, other):
        return _pow(self, other)

    def __rpow__(self, other):
        return _rpow(self, other)

    def __len__(self):
        return self.shape


def dot(left: Union[SymVec, np.ndarray], right: Union[SymVec, np.ndarray]) -> Union[Symbol, SymVec]:
    """
     dot product between symbol vector and symbol vector or symbol vector and np.ndarray
     :param left
     :type: Union[SymVec, np.ndarray]
     :param right
     :type: Union[SymVec, np.ndarray]
     :return: Symbol vector or symbol
     :rtype: Union[SymVec, Symbol]
    """
    result = 0
    # if dot product of SymVec with SymVec, same as mul then sum
    if isinstance(left, SymVec) and isinstance(right, SymVec):
        if left.shape == right.shape:
            return (left * right).sum()
        else:
            raise IndexError('Dimension incorrect')
    elif isinstance(left, np.ndarray):
        if left.ndim == 1:  # if left is a np vector, it should return a Symbol
            for i in range(left.shape[0]):
                result = result + left[i] * right.symbols[i]
        elif left.ndim == 2:  # if left is a np matrix, it should return a SymVec
            for i in range(left.shape[1]):
                result = SymVec([dot(left[i], right)
                                for i in range(left.shape[0])], names=right.names)
        else:
            raise NotImplementedError
    # if right is np array, it should only be vecotr
    elif isinstance(left, SymVec) and isinstance(right, np.ndarray):
        if left.shape == right.shape[0] and right.ndim == 1:
            for i in range(left.shape):
                result = result + left.symbols[i] * right[i]
        else:
            raise IndexError('Dimension incorrect')
    else:
        raise TypeError('Illegal Type')
    return result
