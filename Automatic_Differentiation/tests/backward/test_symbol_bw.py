"""
Testing suite for symbols
"""

import sys
import pytest
import numpy as np

sys.path.append('src/')
sys.path.append('../../src')

from autodiff.backward.sym import Symbol
from autodiff.backward import symbol

# Symbol:
def test_symbol():
    assert isinstance(symbol('a b')[0], Symbol) and symbol(
        'a b')[0].name == 'a'
    assert isinstance(symbol('a b')[1], Symbol) and symbol(
        'a b')[1].name == 'b'


def test_init_symbol():
    a, b = symbol('a b')
    point = {'a': 1, 'b': 2}
    assert a.constant == b.constant is None
    assert a.name == 'a'
    assert b.name == 'b'
    assert a.left_symbol == a.right_symbol == b.left_symbol == b.right_symbol is None
    assert a.op == b.op is None


def test_deriv_symbol():
    a, b, c = symbol('a b c')
    point = {'a':1, 'b': 2, 'c':3}
    seed = {'a':2, 'b': 5, 'c':1}
    f = b ** 2 + a ** 2 - c
    # With seed
    assert f.deriv(point, seed) == 20 + 4 - 1
    # Without seed
    assert f.deriv(point) == {'a': 2, 'b': 4, 'c': -1}


def test_init_symbol_fail():
    with pytest.raises(Exception):
        a, b = symbol('a a')


def test_init_symbol_fail_1():
    with pytest.raises(Exception):
        a, b = symbol('')


def test_init_symbol_fail_2():
    with pytest.raises(Exception):
        a = Symbol(constant=1, name='a')


def test_eval_symbol():
    a, b = symbol('a b')
    point = {'a': 1, 'b': 2}
    f = a + b
    assert f.eval(point) == 3


def test_eval_symbol_fail_2():
    with pytest.raises(Exception):
        a, b = symbol('a b')
        f = a + b
        point = {'a': 1, 'c': 2}
        f.eval(point)


def test_add_symbol():
    a, b = symbol('a b')
    f = 2 * a + b + 1
    point = {'a': 1, 'b': 2}

    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 5
    assert deriv['a'] == 2
    assert deriv['b'] == 1


def test_radd_symbol():
    a, b = symbol('a b')
    f = 1 + 2 * a + b
    point = {'a': 1, 'b': 2}

    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 5
    assert deriv['a'] == 2
    assert deriv['b'] == 1


def test_mult_symbol():
    a, b = symbol('a b')
    f = a * b + 2 * a
    point = {'a': 1, 'b': 2}

    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 4
    assert deriv['a'] == 4
    assert deriv['b'] == 1


def test_divide_symbol():
    a, b = symbol('a b')
    f = a * b / 1 + 3 * b / a + 1 / a
    point = {'a': 1, 'b': 2}
    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 9
    assert deriv['a'] == -5
    assert deriv['b'] == 4


def test_pow_symbol():
    a, b = symbol('a b')
    f = a ** 0.5 + b ** 1
    point = {'a': 1, 'b': 2}
    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 3
    assert deriv['a'] == 0.5
    assert deriv['b'] == 1


def test_pow_symbol_1():
    a, b = symbol('a b')
    f = a ** b
    point = {'a': 2, 'b': 3}
    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 8
    assert deriv['a'] == 12
    assert np.isclose(deriv['b'], 8 * np.log(2))


def test_rpow_symbol():
    a, b = symbol('a b')
    f = 3 ** a + 2 ** b
    point = {'a': -2, 'b': 4}
    result = f.eval(point)
    deriv = f.deriv(point)

    assert np.isclose(result, 1 / 9 + 16)
    assert np.isclose(deriv['a'], 1 / 9 * np.log(3))
    assert np.isclose(deriv['b'], 16 * np.log(2))


def test_mpow_symbol():
    a = symbol('a')
    f1 = a + 2
    f2 = a * 2
    f = f1 ** f2
    point = {'a': 1}
    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 9
    assert np.isclose(deriv['a'], 9 * (2 * np.log(3) + 2 / 3))


def test_sub_symbol():
    a, b = symbol('a b')
    f = b - 0.5 * a - 1
    point = {'a': 1, 'b': 2}
    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == 0.5
    assert deriv['a'] == -0.5
    assert deriv['b'] == 1


def test_neg_symbol():
    a, b = symbol('a b')
    f = -b - 0.5 * a - 5
    point = {'a': 1, 'b': 2}
    result = f.eval(point)
    deriv = f.deriv(point)

    assert result == -7.5
    assert deriv['a'] == -0.5
    assert deriv['b'] == -1
