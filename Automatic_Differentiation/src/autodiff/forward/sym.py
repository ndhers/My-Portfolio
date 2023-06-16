"""
The core object of automatic differentiation, representing variables used in
an AD computation. This implementation targets forward mode AD specifically.
For a reverse mode version, see the submodule `autodiff.backward`
"""

from typing import Callable, List, Union
import numpy as np
NumberTypes = (int, float, complex)


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


def symbolic(op: Callable[['DualNumber', Union['DualNumber', None]], 'DualNumber']) -> Callable[['Symbol', Union['Symbol', None]], 'Symbol']:
    """Decorator that converts a function of dual numbers into a function of
    Symbols, building a computational graph. This function is not intended
    for general use. Use only if you would like to define your own custom
    functions on Symbols.

    :param op: function that takes DualNumbers are arguments, and returns a
        DualNumber carrying the primal and dual traces of the computation
    :type op: Callable[['DualNumber', Union['DualNumber', None]], 'DualNumber']
    """
    def symbolic_func(left_symbol: 'Symbol', right_symbol: 'Symbol' = None):
        return Symbol(
            left=left_symbol,
            right=right_symbol,
            op=op,
            names=left_symbol.names.union(
                right_symbol.names) if right_symbol is not None else left_symbol.names
        )

    return symbolic_func


@symbolic
def _add(a: 'DualNumber', b: 'DualNumber') -> 'DualNumber':
    return a + b


@symbolic
def _multiply(a: 'DualNumber', b: 'DualNumber') -> 'DualNumber':
    return a * b


@symbolic
def _divide(a: 'DualNumber', b: 'DualNumber') -> 'DualNumber':
    real = a.real / b.real
    dual = (b.real * a.dual - a.real * b.dual) / (b.real ** 2)
    return DualNumber(real, dual)


def _pow(a: 'DualNumber', n: float) -> 'DualNumber':
    @symbolic
    def _pow_n(a: 'DualNumber') -> 'DualNumber':
        real = a.real ** n
        dual = a.dual * n * (a.real ** (n - 1))  # power rule
        return DualNumber(real, dual)

    return _pow_n(a)


def _rpow(n: float, a: 'DualNumber') -> 'DualNumber':
    @symbolic
    def _rpow_n(a: 'DualNumber') -> 'DualNumber':
        real = n ** a.real
        dual = n ** a.real * np.log(n) * a.dual
        return DualNumber(real, dual)

    return _rpow_n(a)


@symbolic
def _mpow(a: 'DualNumber', b: 'DualNumber') -> 'DualNumber':
    real = a.real ** b.real
    dual = real * (b.dual * np.log(a.real) + b.real * a.dual / a.real)
    return DualNumber(real, dual)


class Symbol:
    def __init__(self, constant: float = None, name: str = None, left: 'Symbol' = None, right: 'Symbol' = None, op=None,
                 names=None) -> None:
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
        :param names: track names of children symbols
        :type names: List[str]
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
        self.op = op
        # for vec tracing
        if name is not None:
            self.names = {name}
        else:
            self.names = names if names is not None else set()

    def eval(self, input: dict) -> dict:
        """Evaluate a Symbol with at a particular input.

        :param input: the point at which to evaluate the Symbol. The dict should
            have keys that are variable names, and values that are the corresponding
            variable values
        :type input: dict
        :return: a dictionary mapping variable names to resulting values
        :rtype: dict
        """

        first_var = next(iter(input))
        trace = self._trace(with_respect_to=first_var, input=input)
        return trace.real

    def deriv(self, input: dict, seed: dict = None) -> Union[dict, float]:
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

        all_deriv = {}
        for var in input:
            trace = self._trace(var, input)
            all_deriv[var] = trace.dual

        if seed:
            dir_deriv = 0
            for var in seed:
                dir_deriv += seed[var] * all_deriv[var]
            return dir_deriv

        return all_deriv

    def _trace(self, with_respect_to: str, input: dict) -> 'DualNumber':
        if self.left_symbol == None and self.right_symbol == None:
            if self.name == None:
                return DualNumber(self.constant, 0)
            else:
                if self.name not in input:
                    raise ValueError(
                        f'value for variable "{self.name}" not specified')

                val = input[self.name]
                deriv = 1 if self.name == with_respect_to else 0
                return DualNumber(val, deriv)
        else:
            ldual = self.left_symbol._trace(with_respect_to, input)

            if self.right_symbol == None:
                return self.op(ldual)
            else:
                rdual = self.right_symbol._trace(with_respect_to, input)
                return self.op(ldual, rdual)

    def __add__(self, other):
        if not isinstance(other, Symbol):
            if not isinstance(other, NumberTypes):
                return NotImplemented
            other = Symbol(constant=other)

        return _add(self, other)

    def __radd__(self, other):
        other = Symbol(constant=other)
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Symbol):
            if not isinstance(other, NumberTypes):
                return NotImplemented
            other = Symbol(constant=other)

        return _add(self, -other)

    def __rsub__(self, other):
        other = Symbol(constant=other)
        return other - self

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if not isinstance(other, Symbol):
            if not isinstance(other, NumberTypes):
                return NotImplemented
            other = Symbol(constant=other)

        return _multiply(self, other)

    def __rmul__(self, other):
        other = Symbol(constant=other)
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, Symbol):
            if not isinstance(other, NumberTypes):
                return NotImplemented
            other = Symbol(constant=other)

        return _divide(self, other)

    def __rtruediv__(self, other):
        other = Symbol(constant=other)
        return other / self

    def __pow__(self, n):
        # operation between symbols
        if isinstance(n, Symbol):
            return _mpow(self, n)
        if not isinstance(n, NumberTypes):
            return NotImplemented
        return _pow(self, n)

    def __rpow__(self, n):
        return _rpow(n, self)


class DualNumber:
    """Simple DualNumber object, intended to organize the primal and dual traces
    for forward mode automatic differentiation. This object is not intended
    for direct use, and should only be used for advanced customizations.
    """

    def __init__(self, real: float = 0, dual: float = 0):
        """Initiate a new dual number

        :param real: the real part of the dual number, defaults to 0
        :type real: float, optional
        :param dual: the dual part of the dual number, defaults to 0
        :type dual: float, optional
        """

        self.real = real
        self.dual = dual

    def __add__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(other, 0)

        return DualNumber(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        other = DualNumber(other, 0)
        return self + other

    def __mul__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(other, 0)

        real = self.real * other.real
        dual = self.real * other.dual + other.real * self.dual
        return DualNumber(real, dual)

    def __rmul__(self, other):
        other = DualNumber(other, 0)
        return self * other

    def __str__(self):
        return f'{self.real} + {self.dual}e'

    def __repr__(self):
        return str(self)
