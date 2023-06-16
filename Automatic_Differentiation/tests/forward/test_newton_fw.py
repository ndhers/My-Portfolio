import sys
import pytest
import numpy as np
import cmath

sys.path.append('src/')
sys.path.append('../../src')
from autodiff.forward import symbol


def newton_root(f, point):
	error = f.eval(point)
	tol = 1e-15
	iteration = 0
	while error > tol:
		point['x'] = point['x'] - f.eval(point)/f.deriv(point)['x']
		error = f.eval(point)
		iteration += 1
	return point


def test_newton_root():
	x = symbol('x')
	f = 3 * x ** 2 + 2 * x / 3 - 2 # example function
	sol1 = (-2/3+cmath.sqrt((2/3)**2-4*3*(-2)))/(2*3) # positive root
	point = {'x': 2} # initial point
	assert np.isclose(newton_root(f, point)['x'], sol1)

