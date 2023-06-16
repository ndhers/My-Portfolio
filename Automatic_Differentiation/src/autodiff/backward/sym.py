"""
The core object of automatic differentiation, representing variables used in
an AD computation. This implementation targets reverse mode AD specifically.
For the forward mode version, see `autodiff.forward`.
"""

from typing import Callable, List, Tuple, Union
import numpy as np


def symbol(expr: str) -> Union['Symbol', List['Symbol']]:
    """Produce a symbol or tuple of symbols parsed from the provided string.
    Tuples are inferred through spaces in the string. Variable names should
    be delimited by spaces. For example

        a, b = symbol('a b')

    will produce 2 symbols, named 'a' and 'b' respectively. If only one variable
    name is provided, only 1 symbol will be generated. Otherwise, a tuple of
    symbols will be returned.

    This method is the recommended way to make a Symbol.


    :param expr: space-delimited string of variable names
    :type expr: str
    :raises ValueError: if duplicate names are present in the expr string, this
        function will raise an error
    :return: tuple or individual symbol with the given variable names
    :rtype: Union['Symbol', List['Symbol']]
    """

    var_names = expr.split()

    if len(set(var_names)) != len(var_names):
        raise ValueError('duplicate variable names not allowed: ', expr)

    symbols = [Symbol(name=name) for name in var_names]

    if len(symbols) == 1:
        return symbols[0]

    return symbols


def symbolic(op: Callable[[float, Union[float, None]], float]) -> Callable[['Symbol', Union['Symbol', None]], 'Symbol']:
    """Decorator that converts a function of dual numbers into a function of
    Symbols, building a computational graph. This function is not intended
    for general use. Use only if you would like to define your own custom
    functions on Symbols.

    :param op: function that takes DualNumbers are arguments, and returns a
        DualNumber carrying the primal and dual traces of the computation
    :type op: Callable[[float, Union[float, None]], float]
    """

    def symbolic_func(left_symbol: 'Symbol', right_symbol: 'Symbol' = None):
        return Symbol(
            left=left_symbol,
            right=right_symbol,
            op=op
        )

    return symbolic_func


@symbolic
def _add(a: float, b: float, deriv=False) -> Union[float, Tuple]:
    if deriv:
        return (1, 1)
    else:
        return a + b


@symbolic
def _multiply(a: float, b: float, deriv=False) -> Union[float, Tuple]:
    if deriv:
        return (b, a)
    return a * b


@symbolic
def _divide(a: float, b: float, deriv=False) -> Union[float, Tuple]:
    if deriv:
        return (1 / b, -a / (b ** 2))
    return a / b


def _pow(a: float, n: float) -> float:

    @symbolic
    def _pow_n(a: float, deriv=False) -> float:
        if deriv:
            return n * a ** (n - 1)
        return a ** n

    return _pow_n(a)


def _rpow(n: float, a: float) -> float:

    @symbolic
    def _rpow_n(a: float, deriv=False) -> float:
        if deriv:
            return n ** a * np.log(n)
        return n ** a

    return _rpow_n(a)


@symbolic
def _mpow(a: float, b: float, deriv=False) -> float:
    if deriv:
        return (b*a**(b-1), a**b*np.log(a))
    return a**b


class Symbol:
    def __init__(self, constant: float = None, name: str = None, left: 'Symbol' = None, right: 'Symbol' = None, op=None) -> None:
        """Create a new Symbol object. Note, only advanced users with very
        specific applications in mind should construct a Symbol object directly.
        In the vast majority of cases, Symbols should be constructed through the
        factory method `symbol()`.

        :param constant: if this Symbol is a constant rather than a variable, 
            specify the constant value here. Defaults to None
        :type constant: float, optional
        :param name: if this Symbol is a variable, specify the variable name here.
            All Symbols elementary Symbols (Sybmols with no children) should be
            either a variable or a constant. Defaults to None
        :type name: str, optional
        :param left: left child of the Symbol in the computational graph. If the
            Symbol only has a single child, it should be a left child. Defaults to None
        :type left: Symbol, optional
        :param right: right child of the Symbol, defaults to None
        :type right: Symbol, optional
        :param op: function that should be applied to the Symbols children on
            order to obtain the Symbol's value, defaults to None
        :type op: Callable, optional
        :raises ValueError: raised if a Symbol is declared both a constant and
            a variable
        :raises ValueError: raised if an elementary Symbol is not declared as
            a constant or variable
        """
        if constant != None and name != None:
            raise ValueError(
                'cannot declare a symbol both a constant and a variable')

        if constant == None and name == None and left == None:
            raise ValueError(
                'must declare single symbol as either a constant or variable')

        self.constant = constant
        self.name = name
        self.left_symbol = left
        self.right_symbol = right

        self.left_deriv = 0
        self.right_deriv = 0
        self.total_deriv = 0

        self.parent_iter = 0

        self.op = op

    def _forward(self, input: dict) -> float:
        if self.left_symbol == None and self.right_symbol == None:
            if self.name == None:
                return self.constant
            else:
                if self.name not in input:
                    raise ValueError(
                        f'value for variable "{self.name}" not specified')

                val = input[self.name]
                return val
        else:
            left_val = self.left_symbol._forward(input)
            self.left_symbol.parent_iter += 1

            if self.right_symbol == None:
                self.left_deriv = self.op(left_val, deriv=True)
                return self.op(left_val)
            else:
                right_val = self.right_symbol._forward(input)
                self.right_symbol.parent_iter += 1
                self.left_deriv, self.right_deriv = self.op(
                    left_val, right_val, deriv=True)

                return self.op(left_val, right_val)

    def _backward(self, root=False, all_derivs={}):
        if root:
            self.total_deriv = 1

        if self.parent_iter != 0:
            return all_derivs

        if self.left_symbol == None and self.right_symbol == None:
            if self.name != None:
                all_derivs[self.name] = self.total_deriv
            return all_derivs
        else:
            self.left_symbol.total_deriv += self.left_deriv * self.total_deriv
            self.left_symbol.parent_iter -= 1
            all_derivs = self.left_symbol._backward(all_derivs=all_derivs)

            if self.right_symbol != None:
                self.right_symbol.total_deriv += self.right_deriv * self.total_deriv
                self.right_symbol.parent_iter -= 1
                all_derivs = self.right_symbol._backward(all_derivs=all_derivs)

            return all_derivs

    def eval(self, input: dict) -> dict:
        """Evaluate a Symbol with at a particular input.

        :param input: the point at which to evaluate the Symbol. The dict should
            have keys that are variable names, and values that are the corresponding
            variable values
        :type input: dict
        :return: a dictionary mapping variable names to resulting values
        :rtype: dict
        """
        return self._forward(input)

    def deriv(self, input: dict, seed: dict = None) -> dict:
        """Take the derivative of a Symbol with respect to the particular
        input and seed value.

        :param input: dictionary mapping variable names to their corresponding
            input values
        :type input: dict
        :param seed: dictionary mapping variable names to their corresponding
            weights. If the weights normalize to one, then this corresponds to
            requesting a directional derivative in the direction of this seed
            value.
        :type seed: dict, optional
        :return: if no seed is specified, then a dictionary mapping variable
            names to derivatives is returned. If a seed is specified, then
            a float is returned representing the directional/weighted derivative
        :rtype: Union[dict, float]
        """
        self.zero()
        self._forward(input)
        all_deriv = self._backward(root=True)

        if seed:
            dir_deriv = 0
            for var in seed:
                dir_deriv += seed[var] * all_deriv[var]
            return dir_deriv

        return all_deriv

    def zero(self) -> None:
        """Zero all derivatives and parent reference counters on this symbol.
        """
        self.total_deriv = 0
        self.left_deriv = 0
        self.right_deriv = 0
        self.parent_iter = 0

        if self.left_symbol:
            self.left_symbol.zero()
        if self.right_symbol:
            self.right_symbol.zero()

    def __add__(self, other):
        if not isinstance(other, Symbol):
            other = Symbol(constant=other)

        return _add(self, other)

    def __radd__(self, other):
        other = Symbol(constant=other)
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Symbol):
            other = Symbol(constant=other)

        return _add(self, -other)

    def __rsub__(self, other):
        other = Symbol(constant=other)
        return other - self

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if not isinstance(other, Symbol):
            other = Symbol(constant=other)

        return _multiply(self, other)

    def __rmul__(self, other):
        other = Symbol(constant=other)
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, Symbol):
            other = Symbol(constant=other)

        return _divide(self, other)

    def __rtruediv__(self, other):
        other = Symbol(constant=other)
        return other / self

    def __pow__(self, n):
        # operation between symbols
        if isinstance(n, Symbol):
            return _mpow(self, n)

        return _pow(self, n)

    def __rpow__(self, n):
        return _rpow(n, self)
