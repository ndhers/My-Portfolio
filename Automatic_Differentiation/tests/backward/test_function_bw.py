"""
Testing suite for function
"""
import math
import sys

import numpy as np
import pytest

sys.path.append('src/')
sys.path.append('../../src')

from autodiff.backward import symbol
import autodiff.backward as bn


def test_sum():
    a = symbol('a')
    f1 = a ** 2
    f2 = 3 * a
    f3 = bn.log(a)
    f = bn.sum([f1, f2, f3, 3])
    point = {'a': 2}
    result = f.eval(point)
    deriv = f.deriv(point)
    assert result == 10 + np.log(2) + 3
    assert deriv['a'] == 7.5


def test_sum_2():
    a = symbol('a')
    f1 = bn.log(a)
    f = bn.sum([f1])
    point = {'a': 2}
    result = f.eval(point)
    deriv = f.deriv(point)
    assert np.isclose(result, np.log(2))
    assert deriv['a'] == 0.5


def test_prod():
    a = symbol('a')
    f1 = a ** 2
    f2 = 3 * a
    f3 = bn.log(a)
    f = bn.prod([f1, f2, f3, 3])
    point = {'a': 2}
    result = f.eval(point)
    deriv = f.deriv(point)
    assert result == 24 * np.log(2) * 3
    assert deriv['a'] == (4 * 6 * np.log(2) + 4 * 3 * np.log(2) + 4 * 6 * 1 / 2) * 3


def test_prod_2():
    a = symbol('a')
    f1 = a ** 3
    f = bn.prod([f1])
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
    b = symbol('b')
    m = bn.sin(b)
    point = {'b': np.pi / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert val == 1
    assert np.round(der, 6) == 0


def test_cos():
    b = symbol('b')
    m = bn.cos(b)
    point = {'b': np.pi / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.round(val, 6) == 0
    assert der == -1


def test_tan():
    b = symbol('b')
    m = bn.tan(b)
    point = {'b': np.pi}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, 0)
    assert np.isclose(der, 1)


def test_arcsin():
    b = symbol('b')
    m = bn.arcsin(b)
    point = {'b': 1 / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, np.pi / 6)
    assert np.isclose(der, 2 / 3 ** 0.5)


def test_arccos():
    b = symbol('b')
    m = bn.arccos(b)
    point = {'b': 1 / 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, np.pi / 3)
    assert np.isclose(der, -2 / 3 ** 0.5)


def test_arctan():
    b = symbol('b')
    m = bn.arctan(b)
    point = {'b': 1}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, np.pi / 4)
    assert np.isclose(der, 1 / 2)


# Hyperbolic functions

def test_sinh():
    b = symbol('b')
    m = bn.sinh(b)
    point = {'b': 0}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, 0)
    assert np.isclose(der, 1)


def test_cosh():
    b = symbol('b')
    m = bn.cosh(b)
    point = {'b': 0}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, 1)
    assert np.isclose(der, 0)


def test_tanh():
    b = symbol('b')
    m = bn.tanh(b)
    point = {'b': 0}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, 0)
    assert np.isclose(der, 1)


# Natural functions

def test_sqrt():
    b = symbol('b')
    m = bn.sqrt(b)
    point = {'b': 4}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, np.sqrt(4))
    assert np.isclose(der, 1 / (2 * np.sqrt(4)))


def test_exp():
    b = symbol('b')
    m = bn.exp(b)
    point = {'b': 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, np.exp(2))
    assert np.isclose(der, np.exp(2))


def test_log():
    b = symbol('b')
    m = bn.log(b)
    point = {'b': 2}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, math.log(2))
    assert np.isclose(der, 1 / 2)


def test_log10():
    b = symbol('b')
    m = bn.log10(b)
    point = {'b': 100}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, 2)
    assert np.isclose(der, 1 / (100 * np.log(10)))


def test_logbase():
    a = symbol('a')
    m = bn.log_base(a, 4)
    point = {'a': 16}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1})
    assert np.isclose(val, 2)
    assert np.isclose(der, 1 / (16 * np.log(4)))


def test_logbase_2():
    a, b = symbol('a b')
    m = bn.log_base(a, base=b)
    point = {'a': 16, 'b':2}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 4)
    assert np.isclose(der, 1 / (16 * np.log(2)) - np.log(16)/(2 * np.log(2)**2))


def test_logbase_fail():
    with pytest.raises(Exception):
        a, b = symbol('a b')
        m = bn.log_base(a, -1) + b ** 2


# def test_logbase_fail_2():
#     with pytest.raises(Exception):
#         a, b = symbol('a b')
#         m = bn.log_base(a, b)


# Activation functions


def test_sigmoid():
    b = symbol('b')
    m = bn.sigmoid(b)
    point = {'b': 1}
    val = m.eval(point)
    der = m.deriv(point, seed={'b': 1})
    assert np.isclose(val, 1 / (np.exp(-1) + 1))
    assert np.isclose(der, 1 / (np.exp(-1) + 1) * (1 - (1 / (np.exp(-1) + 1))))


def test_ReLU():
    a, b = symbol('a b')
    m = bn.ReLU(a) + bn.ReLU(b)
    point = {'a': 3, 'b': -1}
    val = m.eval(point)
    der = m.deriv(point, seed={'a': 1, 'b': 1})
    assert np.isclose(val, 3)
    assert np.isclose(der, 1)
