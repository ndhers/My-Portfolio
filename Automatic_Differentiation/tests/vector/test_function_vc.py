"""
Testing suite for function
"""
import math
import sys

import numpy as np
import pytest

sys.path.append('src/')
sys.path.append('../../src')

import autodiff.forward as fw
from autodiff.vector import SymVec
import autodiff.vector as vc


# trigonometric functions

def test_sin():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.sin(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.sin(1), np.sin(2), np.sin(3)])))
    assert np.all(np.isclose(der, np.array([[np.cos(1), 0, 0], [0, np.cos(2), 0], [0, 0, np.cos(3)]])))


def test_cos():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.cos(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.cos(1), np.cos(2), np.cos(3)])))
    assert np.all(np.isclose(der, np.array([[- np.sin(1), 0, 0], [0, - np.sin(2), 0], [0, 0, - np.sin(3)]])))


def test_tan():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.tan(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.tan(1), np.tan(2), np.tan(3)])))
    assert np.all(
        np.isclose(der, np.array([[np.tan(1) ** 2 + 1, 0, 0], [0, np.tan(2) ** 2 + 1, 0], [0, 0, np.tan(3) ** 2 + 1]])))


def test_arcsin():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.arcsin(v1)
    point = {'a': 0.1, 'b': 0.2, 'c': 0.3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.arcsin(0.1), np.arcsin(0.2), np.arcsin(0.3)])))
    assert np.all(np.isclose(der, np.array(
        [[1 / np.sqrt(1 - 0.1 ** 2), 0, 0], [0, 1 / np.sqrt(1 - 0.2 ** 2), 0], [0, 0, 1 / np.sqrt(1 - 0.3 ** 2)]])))


def test_arccos():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.arccos(v1)
    point = {'a': 0.1, 'b': 0.2, 'c': 0.3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.arccos(0.1), np.arccos(0.2), np.arccos(0.3)])))
    assert np.all(np.isclose(der, np.array([[- 1 / np.sqrt(1 - 0.1 ** 2), 0, 0], [0, - 1 / np.sqrt(1 - 0.2 ** 2), 0],
                                            [0, 0, - 1 / np.sqrt(1 - 0.3 ** 2)]])))


def test_arctan():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.arctan(v1)
    point = {'a': 0.1, 'b': 0.2, 'c': 0.3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.arctan(0.1), np.arctan(0.2), np.arctan(0.3)])))
    assert np.all(
        np.isclose(der, np.array([[1 / (1 + 0.1 ** 2), 0, 0], [0, 1 / (1 + 0.2 ** 2), 0], [0, 0, 1 / (1 + 0.3 ** 2)]])))


# Hyperbolic functions

def test_sinh():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.sinh(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.sinh(1), np.sinh(2), np.sinh(3)])))
    assert np.all(np.isclose(der, np.array([[np.cosh(1), 0, 0], [0, np.cosh(2), 0], [0, 0, np.cosh(3)]])))


def test_cosh():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.cosh(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.cosh(1), np.cosh(2), np.cosh(3)])))
    assert np.all(np.isclose(der, np.array([[np.sinh(1), 0, 0], [0, np.sinh(2), 0], [0, 0, np.sinh(3)]])))


def test_tanh():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.tanh(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([np.tanh(1), np.tanh(2), np.tanh(3)])))
    assert np.all(np.isclose(der, np.array(
        [[(1 / np.cosh(1)) ** 2, 0, 0], [0, (1 / np.cosh(2)) ** 2, 0], [0, 0, (1 / np.cosh(3)) ** 2]])))


# Natural functions


def test_sqrt():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.sqrt(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.sqrt([1, 2, 3])))
    assert np.all(np.isclose(der, np.array(
        [[1 / (2 * np.sqrt(1)), 0, 0], [0, 1 / (2 * np.sqrt(2)), 0], [0, 0, 1 / (2 * np.sqrt(3))]])))


def test_exp():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.exp(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.exp([1, 2, 3])))
    assert np.all(np.isclose(der, np.array([[np.exp(1), 0, 0], [0, np.exp(2), 0], [0, 0, np.exp(3)]])))


def test_log():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.log(v1)
    point = {'a': 2, 'b': 3, 'c': 4}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.log([2, 3, 4])))
    assert np.all(np.isclose(der, np.array(
        [[1 / 2 * np.log(math.e), 0, 0], [0, 1 / 3 * np.log(math.e), 0], [0, 0, 1 / 4 * np.log(math.e)]])))


def test_log_base():
    a, b, c, d, e, f = fw.symbol('a b c d e f')
    v1 = vc.concat([a, b, c])
    v2 = vc.concat([d, e, f])
    fv = vc.log_base(v1, v2)
    point = {'a': 4, 'b': 9, 'c': 16, 'd': 2, 'e': 3, 'f': 4}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([2, 2, 2])))
    assert np.all(np.isclose(der, np.array([[1 / (np.log(2) * 4), 0, 0, -np.log(4) / (2 * np.log(2) ** 2), 0, 0],
                                            [0, 1 / (np.log(3) * 9), 0, 0, -np.log(9) / (3 * np.log(3) ** 2), 0],
                                            [0, 0, 1 / (np.log(4) * 16), 0, 0, -np.log(16) / (4 * np.log(4) ** 2)]])))


def test_log10():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.log10(v1)
    point = {'a': 2, 'b': 3, 'c': 4}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([math.log10(2), math.log10(3), math.log10(4)])))
    assert np.all(np.isclose(der, np.array(
        [[1 / (2 * np.log(10)), 0, 0], [0, 1 / (3 * np.log(10)), 0], [0, 0, 1 / (4 * np.log(10))]])))


# Activation functions


def test_sigmoid():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.sigmoid(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(
        np.isclose(val, np.array([(np.exp(1)) / (np.exp(1) + 1), 1 / (1 + math.e ** -2), 1 / (1 + math.e ** -3)])))
    assert np.all(np.isclose(der, np.array(
        [[(1 / np.exp(1)) / (1 + (1 / np.exp(1))) ** 2, 0, 0], [0, math.e ** -2 / (1 + math.e ** -2) ** 2, 0],
         [0, 0, math.e ** -3 / (1 + math.e ** -3) ** 2]])))


def test_ReLU():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a, b, c])
    fv = vc.ReLU(v1)
    point = {'a': 1, 'b': 2, 'c': 3}
    val = fv.eval(point)
    der = fv.deriv(point)
    assert np.all(np.isclose(val, np.array([1, 2, 3])))
    assert np.all(np.isclose(der, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
