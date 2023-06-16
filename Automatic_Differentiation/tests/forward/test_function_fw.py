"""
Testing suite for function
"""
import math
import sys

import numpy as np
import pytest

sys.path.append('src/')
sys.path.append('../../src')

from autodiff.forward import symbol
import autodiff.forward as fn


def test_sum():
    a = symbol('a')
    f1 = a ** 2
    f2 = 3 * a
    f3 = fn.log(a)
    f = fn.sum([f1, f2, f3, 3])
    point = {'a': 2}
    result = f.eval(point)
    deriv = f.deriv(point)
    assert result == 10 + np.log(2) + 3
    assert deriv['a'] == 7.5


def test_sum_2():
    a = symbol('a')
    f1 = fn.log(a)
    f = fn.sum([f1])
    point = {'a': 2}
    result = f.eval(point)
    deriv = f.deriv(point)
    assert np.isclose(result, np.log(2))
    assert deriv['a'] == 0.5


def test_prod():
    a = symbol('a')
    f1 = a ** 2
    f2 = 3 * a
    f3 = fn.log(a)
    f = fn.prod([f1, f2, f3, 3])
    point = {'a': 2}
    result = f.eval(point)
    deriv = f.deriv(point)
    assert result == 24 * np.log(2) * 3
    assert deriv['a'] == (4 * 6 * np.log(2) + 4 * 3 * np.log(2) + 4 * 6 * 1 / 2) * 3


def test_prod_2():
    a = symbol('a')
    f1 = a ** 3
    f = fn.prod([f1])
    point = {'a': 2}
    result = f.eval(point)
    deriv = f.deriv(point)
    assert result == 8
    assert deriv['a'] == 12


def test_seed():
    c = symbol('c')
    nc = 1 - c
    der = nc.deriv(input={'c': 1}, seed={'c': 1})
    assert der == -1


# trigonometric functions

def test_sin():
    a, b = symbol('a b')
    m = fn.sin(b)
    point = {'a': 3, 'b': np.pi / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert val == 1
    assert np.round(der, 6) == 0


def test_cos():
    a, b = symbol('a b')
    m = fn.cos(b)
    point = {'a': 3, 'b': np.pi / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.round(val, 6) == 0
    assert der == -1


def test_tan():
    a, b = symbol('a b')
    m = fn.tan(b)
    point = {'a': 3, 'b': np.pi}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 0)
    assert np.isclose(der, 1)


def test_arcsin():
    a, b = symbol('a b')
    m = fn.arcsin(b)
    point = {'a': 3, 'b': 1 / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, np.pi / 6)
    assert np.isclose(der, 2 / 3 ** 0.5)


def test_arccos():
    a, b = symbol('a b')
    m = fn.arccos(b)
    point = {'a': 3, 'b': 1 / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, np.pi / 3)
    assert np.isclose(der, -2 / 3 ** 0.5)


def test_arctan():
    a, b = symbol('a b')
    m = fn.arctan(b)
    point = {'a': 3, 'b': 1}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, np.pi / 4)
    assert np.isclose(der, 1 / 2)


# Hyperbolic functions

def test_sinh():
    a, b = symbol('a b')
    m = fn.sinh(b)
    point = {'a': 3, 'b': 0}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 0)
    assert np.isclose(der, 1)


def test_cosh():
    a, b = symbol('a b')
    m = fn.cosh(b)
    point = {'a': 3, 'b': 0}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 1)
    assert np.isclose(der, 0)


def test_tanh():
    a, b = symbol('a b')
    m = fn.tanh(b)
    point = {'a': 3, 'b': 0}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 0)
    assert np.isclose(der, 1)


# Natural functions

def test_sqrt():
    a, b = symbol('a b')
    m = fn.sqrt(b)
    point = {'a': 3, 'b': 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, np.sqrt(2))
    assert np.isclose(der, 1 / (2 * np.sqrt(2)))


def test_exp():
    a, b = symbol('a b')
    m = fn.exp(b)
    point = {'a': 3, 'b': 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, np.exp(2))
    assert np.isclose(der, np.exp(2))


def test_log():
    a, b = symbol('a b')
    m = fn.log(b)
    point = {'a': 3, 'b': 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, math.log(2))
    assert np.isclose(der, 1 / 2)


def test_log10():
    a, b = symbol('a b')
    m = fn.log10(b)
    point = {'a': 3, 'b': 100}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 2)
    assert np.isclose(der, 1 / (100 * math.log(10)))


def test_logbase():
    a, b = symbol('a b')
    m = fn.log_base(a, 4) + b ** 2
    point = {'a': 16, 'b': 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 6)
    assert np.isclose(der, 1 / (16 * np.log(4)) + 4)


def test_logbase_2():
    a, b = symbol('a b')
    m = fn.log_base(a, base=b)
    point = {'a': 16, 'b': 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 4)
    assert np.isclose(der, 1/(16*np.log(2)) - np.log(16)/(2*(np.log(2))**2))


def test_logbase_fail_1():
    with pytest.raises(Exception):
        a, b = symbol('a b')
        m = fn.log_base(a, base=-1) + b ** 2


# def test_logbase_fail_2():
#     with pytest.raises(Exception):
#         a, b = symbol('a b')
#         m = fn.log_base(a, b)


# Activation functions


def test_sigmoid():
    a, b = symbol('a b')
    m = fn.sigmoid(b)
    point = {'a': 3, 'b': 1}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, (np.exp(1)) / (np.exp(1) + 1))
    assert np.isclose(der, (1 / np.exp(1)) / (1 + (1 / np.exp(1))) ** 2)


def test_ReLU():
    a, b = symbol('a b')
    m = fn.ReLU(a) + fn.ReLU(b)
    point = {'a': 3, 'b': -1}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 3)
    assert np.isclose(der, 1)
