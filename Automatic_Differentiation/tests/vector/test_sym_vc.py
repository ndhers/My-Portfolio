"""
Testing suite for sym vector
"""
import sys

import numpy as np
import pytest

sys.path.append('src/')
sys.path.append('../../src')

import autodiff.forward as fw
from autodiff.vector import SymVec
import autodiff.vector as vc


# 0. test initiation methods
def test_concat():
    a, b, c = fw.symbol('a b c')
    d, e, f = fw.symbol('d e f')
    v1 = vc.concat([a, b, c])
    v2 = vc.concat([d, e, f])
    assert isinstance(v1, SymVec)
    assert isinstance(v2, SymVec)
    assert v1.names == ['a', 'b', 'c']
    assert v2.names == ['d', 'e', 'f']
    assert len(v1) == len(v2) == 3
    f = v1 + v2
    assert f.names == ['a', 'b', 'c', 'd', 'e', 'f']
    assert np.all(np.isclose(f.eval({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}), np.array([5, 7, 9])))
    theo_der = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]])
    assert np.all(np.isclose(f.deriv({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}), theo_der))


def test_concat_2():
    a, b, c = fw.symbol('a b c')
    v1 = vc.concat([a ** 2 + b ** 2, a + b, b * c])
    assert isinstance(v1, SymVec)
    assert v1.names == ['a', 'b', 'c']
    assert np.all(np.isclose(v1.eval({'a': 1, 'b': 2, 'c': 3}), np.array([5, 3, 6])))
    theo_der = np.array([[2, 4, 0], [1, 1, 0], [0, 3, 2]])
    assert np.all(np.isclose(v1.deriv({'a': 1, 'b': 2, 'c': 3}), theo_der))


def test_quick_eval_deriv():
    a, b, c = fw.symbol('a b c')
    d, e, f = fw.symbol('d e f')
    v1 = vc.concat([a, b, c])
    v2 = vc.concat([d, e, f])
    f = v1 + v2
    theo_der = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]])
    assert np.all(np.isclose(f.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([5, 7, 9])))
    assert np.all(np.isclose(f.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


def test_vec_gen():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(3)
    assert isinstance(v1, SymVec)
    assert isinstance(v2, SymVec)
    assert v1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert v2.names == ['dum_00003', 'dum_00004', 'dum_00005']
    assert np.all(np.isclose(v1.quickeval(np.array([1, 2, 3])), np.array([1, 2, 3])))
    assert np.all(np.isclose(v2.quickeval(np.array([1, 2, 3])), np.array([1, 2, 3])))
    theo_der = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.all(np.isclose(v1.quickderiv(np.array([1, 2, 3])), theo_der))
    assert np.all(np.isclose(v2.quickderiv(np.array([1, 2, 3])), theo_der))


def test_gen_fail():
    g = vc.vec_gen()
    with pytest.raises(Exception):
        v1 = g.generate(-1)


def test_eval_fail():
    g = vc.vec_gen()
    v1 = g.generate(3)
    with pytest.raises(Exception):
        v1.quickeval(np.array([[1, 2, 3], [4, 5, 6]]))


# 1. tests for add
def test_add_vec_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(3)
    f = v1 + 2 * v2
    theo_der = np.array([[1, 0, 0, 2, 0, 0], [0, 1, 0, 0, 2, 0], [0, 0, 1, 0, 0, 2]])
    assert np.all(np.isclose(f.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([9, 12, 15])))
    assert np.all(np.isclose(f.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


def test_add_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a, b = fw.symbol('a b')
    f1 = v1 + a + b
    assert f1.names == ['a', 'b'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3, 4, 5])), np.array([6, 7, 8])))
    theo_der = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3, 4, 5])), theo_der))


def test_add_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([1, 2, 3])
    f1 = v1 + a1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([2, 4, 6])))
    theo_der = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.all(np.isclose(v1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_add_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = v1 + n1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([4, 5, 6])))
    theo_der = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_radd_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a, b = fw.symbol('a b')
    f1 = a + v1 + b
    assert f1.names == ['a', 'b'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3, 4, 5])), np.array([6, 7, 8])))
    theo_der = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3, 4, 5])), theo_der))


def test_radd_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([1, 2, 3])
    f1 = a1 + v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([2, 4, 6])))
    theo_der = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.all(np.isclose(v1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_radd_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = n1 + v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([4, 5, 6])))
    theo_der = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_add_fail():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(5)
    with pytest.raises(Exception):
        f = v1 + v2
    v3 = np.random.randn(5)
    with pytest.raises(Exception):
        f = v1 + v3
    with pytest.raises(Exception):
        f = v3 + v1
    v4 = np.random.randn(5, 3)
    with pytest.raises(Exception):
        f = v1 + v4
    with pytest.raises(Exception):
        f = v4 + v1


# 2. tests for sub
def test_sub_vec_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(3)
    f = v1 - 2 * v2
    theo_der = np.array([[1, 0, 0, -2, 0, 0], [0, 1, 0, 0, -2, 0], [0, 0, 1, 0, 0, -2]])
    assert np.all(np.isclose(f.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([-7, -8, -9])))
    assert np.all(np.isclose(f.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


def test_sub_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a, b = fw.symbol('a b')
    f1 = v1 - a - b
    assert f1.names == ['a', 'b'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3, 4, 5])), np.array([0, 1, 2])))
    theo_der = np.array([[-1, -1, 1, 0, 0], [-1, -1, 0, 1, 0], [-1, -1, 0, 0, 1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3, 4, 5])), theo_der))


def test_sub_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([1, 2, 3])
    f1 = v1 - a1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([0, 0, 0])))
    theo_der = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_sub_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = v1 - n1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([-2, -1, 0])))
    theo_der = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rsub_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a, b = fw.symbol('a b')
    f1 = a - b - v1
    assert f1.names == ['a', 'b'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3, 4, 5])), np.array([-4, -5, -6])))
    theo_der = np.array([[1, -1, -1, 0, 0], [1, -1, 0, -1, 0], [1, -1, 0, 0, -1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3, 4, 5])), theo_der))


def test_rsub_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([1, 2, 3])
    f1 = a1 - v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([0, 0, 0])))
    theo_der = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rsub_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = n1 - v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([2, 1, 0])))
    theo_der = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_sub_fail():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(5)
    with pytest.raises(Exception):
        f = v1 - v2
    v3 = np.random.randn(5)
    with pytest.raises(Exception):
        f = v1 - v3
    with pytest.raises(Exception):
        f = v3 - v1
    v4 = np.random.randn(5, 3)
    with pytest.raises(Exception):
        f = v1 - v4
    with pytest.raises(Exception):
        f = v4 - v1


# 3. tests for mul
def test_mul_vec_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(3)
    f = v1 * v2
    theo_der = np.array([[4, 0, 0, 1, 0, 0], [0, 5, 0, 0, 2, 0], [0, 0, 6, 0, 0, 3]])
    assert np.all(np.isclose(f.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([4, 10, 18])))
    assert np.all(np.isclose(f.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


def test_mul_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a, b = fw.symbol('a b')
    f1 = v1 * a * b
    assert f1.names == ['a', 'b'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3, 4, 5])), np.array([6, 8, 10])))
    theo_der = np.array([[6, 3, 2, 0, 0], [8, 4, 0, 2, 0], [10, 5, 0, 0, 2]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3, 4, 5])), theo_der))


def test_mul_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([1, 2, 3])
    f1 = v1 * a1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([1, 4, 9])))
    theo_der = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_mul_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = v1 * n1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([3, 6, 9])))
    theo_der = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rmul_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a, b = fw.symbol('a b')
    f1 = a * v1 * b
    assert f1.names == ['a', 'b'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3, 4, 5])), np.array([6, 8, 10])))
    theo_der = np.array([[6, 3, 2, 0, 0], [8, 4, 0, 2, 0], [10, 5, 0, 0, 2]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3, 4, 5])), theo_der))


def test_rmul_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([1, 2, 3])
    f1 = a1 * v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([1, 4, 9])))
    theo_der = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rmul_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = n1 * v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([3, 6, 9])))
    theo_der = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_mul_fail():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(5)
    with pytest.raises(Exception):
        f = v1 * v2
    v3 = np.random.randn(5)
    with pytest.raises(Exception):
        f = v1 * v3
    with pytest.raises(Exception):
        f = v3 * v1
    v4 = np.random.randn(5, 3)
    with pytest.raises(Exception):
        f = v1 * v4
    with pytest.raises(Exception):
        f = v4 * v1


# 4. tests for div
def test_div_vec_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(3)
    f = v1 / v2
    theo_der = np.array([[1 / 4, 0, 0, -1 / 16, 0, 0], [0, 1 / 5, 0, 0, -2 / 25, 0], [0, 0, 1 / 6, 0, 0, -3 / 36]])
    assert np.all(np.isclose(f.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([1 / 4, 2 / 5, 3 / 6])))
    assert np.all(np.isclose(f.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


def test_div_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a = fw.symbol('a')
    f1 = v1 / a
    assert f1.names == ['a'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([5, 2, 3, 4])), np.array([2 / 5, 3 / 5, 4 / 5])))
    theo_der = np.array([[-2 / 25, 1 / 5, 0, 0], [-3 / 25, 0, 1 / 5, 0], [-4 / 25, 0, 0, 1 / 5]])
    assert np.all(np.isclose(f1.quickderiv(np.array([5, 2, 3, 4])), theo_der))


def test_div_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([3, 4, 5])
    f1 = v1 / a1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([1 / 3, 2 / 4, 3 / 5])))
    theo_der = np.array([[1 / 3, 0, 0], [0, 1 / 4, 0], [0, 0, 1 / 5]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_div_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = v1 / n1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([1 / 3, 2 / 3, 1])))
    theo_der = np.array([[1 / 3, 0, 0], [0, 1 / 3, 0], [0, 0, 1 / 3]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rdiv_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a = fw.symbol('a')
    f1 = a / v1
    assert f1.names == ['a'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3, 4])), np.array([1 / 2, 1 / 3, 1 / 4])))
    theo_der = np.array([[1 / 2, -1 / 4, 0, 0], [1 / 3, 0, -1 / 9, 0], [1 / 4, 0, 0, -1 / 16]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3, 4, 5])), theo_der))


def test_rdiv_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([3, 4, 5])
    f1 = a1 / v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([3, 4 / 2, 5 / 3])))
    theo_der = np.array([[-3, 0, 0], [0, -4 / 4, 0], [0, 0, -5 / 9]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rdiv_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = n1 / v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([3, 3 / 2, 1])))
    theo_der = np.array([[-3, 0, 0], [0, -3 / 4, 0], [0, 0, -3 / 9]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_div_fail():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(5)
    with pytest.raises(Exception):
        f = v1 / v2
    v3 = np.random.randn(5)
    with pytest.raises(Exception):
        f = v1 / v3
    with pytest.raises(Exception):
        f = v3 / v1
    v4 = np.random.randn(5, 3)
    with pytest.raises(Exception):
        f = v1 / v4
    with pytest.raises(Exception):
        f = v4 / v1


# 5. tests for pow
def test_pow_vec_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(3)
    f = v1 ** v2
    theo_der = np.array([[4, 0, 0, 0, 0, 0], [0, 5 * 2 ** 4, 0, 0, 2 ** 5 * np.log(2), 0],
                         [0, 0, 6 * 3 ** 5, 0, 0, 3 ** 6 * np.log(3)]])
    assert np.all(np.isclose(f.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([1, 2 ** 5, 3 ** 6])))
    assert np.all(np.isclose(f.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


def test_pow_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a = fw.symbol('a')
    f1 = v1 ** a
    assert f1.names == ['a'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([5, 2, 3, 4])), np.array([2 ** 5, 3 ** 5, 4 ** 5])))
    theo_der = np.array([[2 ** 5 * np.log(2), 5 * 2 ** 4, 0, 0], [3 ** 5 * np.log(3), 0, 5 * 3 ** 4, 0],
                         [4 ** 5 * np.log(4), 0, 0, 5 * 4 ** 4]])
    assert np.all(np.isclose(f1.quickderiv(np.array([5, 2, 3, 4])), theo_der))


def test_pow_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([3, 4, 5])
    f1 = v1 ** a1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([1 ** 3, 2 ** 4, 3 ** 5])))
    theo_der = np.array([[3, 0, 0], [0, 4 * 2 ** 3, 0], [0, 0, 5 * 3 ** 4]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_pow_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = v1 ** n1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([1 ** 3, 2 ** 3, 3 ** 3])))
    theo_der = np.array([[3, 0, 0], [0, 3 * 2 ** 2, 0], [0, 0, 3 * 3 ** 2]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rpow_vec_sym():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a = fw.symbol('a')
    f1 = a ** v1
    assert f1.names == ['a'] + ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([5, 2, 3, 4])), np.array([5 ** 2, 5 ** 3, 5 ** 4])))
    theo_der = np.array([[2 * 5, 5 ** 2 * np.log(5), 0, 0], [3 * 5 ** 2, 0, 5 ** 3 * np.log(5), 0],
                         [4 * 5 ** 3, 0, 0, 5 ** 4 * np.log(5)]])
    assert np.all(np.isclose(f1.quickderiv(np.array([5, 2, 3, 4])), theo_der))


def test_rpow_vec_array():
    g = vc.vec_gen()
    v1 = g.generate(3)
    a1 = np.array([3, 4, 5])
    f1 = a1 ** v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([3 ** 1, 4 ** 2, 5 ** 3])))
    theo_der = np.array([[3 ** 1 * np.log(3), 0, 0], [0, 4 ** 2 * np.log(4), 0], [0, 0, 5 ** 3 * np.log(5)]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_rpow_vec_num():
    g = vc.vec_gen()
    v1 = g.generate(3)
    n1 = 3
    f1 = n1 ** v1
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([3 ** 1, 3 ** 2, 3 ** 3])))
    theo_der = np.array([[3 ** 1 * np.log(3), 0, 0], [0, 3 ** 2 * np.log(3), 0], [0, 0, 3 ** 3 * np.log(3)]])
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_pow_fail():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(5)
    with pytest.raises(Exception):
        f = v1 ** v2
    v3 = np.random.randn(5)
    with pytest.raises(Exception):
        f = v1 ** v3
    with pytest.raises(Exception):
        f = v3 ** v1
    v4 = np.random.randn(5, 3)
    with pytest.raises(Exception):
        f = v1 ** v4
    with pytest.raises(Exception):
        f = v4 ** v1


# 6. tests for neg

def test_neg_vec():
    g = vc.vec_gen()
    v1 = g.generate(6)
    v2 = -v1
    assert v2.names == ['dum_00000', 'dum_00001', 'dum_00002', 'dum_00003', 'dum_00004', 'dum_00005']
    assert np.all(np.isclose(v2.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([-1, -2, -3, -4, -5, -6])))
    theo_der = np.array(
        [[-1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, -1, 0],
         [0, 0, 0, 0, 0, -1]])
    assert np.all(np.isclose(v2.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


# 7. tests for dot product
def test_dot_vec_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(3)
    f = vc.concat([vc.dot(v1, v2)])
    theo_der = np.array([[4, 5, 6, 1, 2, 3]])
    assert np.all(np.isclose(f.quickeval(np.array([1, 2, 3, 4, 5, 6])), np.array([32])))
    assert np.all(np.isclose(f.quickderiv(np.array([1, 2, 3, 4, 5, 6])), theo_der))


def test_dot_vec_arr():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = np.array([1, 2, 3])
    f = vc.concat([vc.dot(v1, v2)])
    theo_der = np.array([[1, 2, 3]])
    assert np.all(np.isclose(f.quickeval(np.array([4, 5, 6])), np.array([32])))
    assert np.all(np.isclose(f.quickderiv(np.array([4, 5, 6])), theo_der))


def test_dot_arr_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = np.array([1, 2, 3])
    f = vc.concat([vc.dot(v2, v1)])
    theo_der = np.array([[1, 2, 3]])
    assert np.all(np.isclose(f.quickeval(np.array([4, 5, 6])), np.array([32])))
    assert np.all(np.isclose(f.quickderiv(np.array([4, 5, 6])), theo_der))


def test_dot_mat_vec():
    g = vc.vec_gen()
    m1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v1 = g.generate(3)
    f1 = vc.dot(m1, v1)
    assert f1.names == ['dum_00000', 'dum_00001', 'dum_00002']
    assert np.all(np.isclose(f1.quickeval(np.array([1, 2, 3])), np.array([14, 32, 50])))
    assert np.all(np.isclose(f1.quickderiv(np.array([1, 2, 3])), m1))


def test_dot_vec_vec_fail_1():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = g.generate(5)
    with pytest.raises(Exception):
        f = vc.concat([vc.dot(v1, v2)])
    with pytest.raises(Exception):
        f = vc.concat([vc.dot(v1, 'fail')])


def test_dot_vec_vec_fail_2():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = np.random.randn(5)
    with pytest.raises(Exception):
        f = vc.concat([vc.dot(v1, v2)])


def test_dot_vec_vec_fail_3():
    g = vc.vec_gen()
    v1 = g.generate(3)
    v2 = np.random.randn(5, 3)
    with pytest.raises(Exception):
        f = vc.concat([vc.dot(v1, v2)])
    v3 = np.random.randn(3, 5)
    with pytest.raises(Exception):
        f = vc.concat([vc.dot(v3, v1)])
    v4 = np.random.randn(3, 4, 5)
    with pytest.raises(Exception):
        f = vc.concat([vc.dot(v4, v1)])


# 8. test sum and prod
def test_sum_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    s1 = vc.concat([v1.sum()])
    assert np.isclose(s1.quickeval(np.array([1, 2, 3])), np.array([6]))
    theo_der = np.array([[1, 1, 1]])
    assert np.all(np.isclose(s1.quickderiv(np.array([1, 2, 3])), theo_der))


def test_prod_vec():
    g = vc.vec_gen()
    v1 = g.generate(3)
    p1 = vc.concat([v1.prod()])
    assert np.isclose(p1.quickeval(np.array([1, 2, 3])), np.array([6]))
    theo_der = np.array([[6, 3, 2]])
    assert np.all(np.isclose(p1.quickderiv(np.array([1, 2, 3])), theo_der))
