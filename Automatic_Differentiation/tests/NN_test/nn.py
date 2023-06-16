import autodiff.forward as fw
import numpy as np

# Simple NN example
d = 784
N = 10
input_ = np.random.randn(d)
# print(input_.ndim)
# layer 1
l1 = 5
l2 = 2
param_val = np.array(
    [np.random.rand() * 1 / d for _ in range(l1 * d)] + [np.random.rand() * 1 / l1 for _ in range(l2 * l1)])


def nn(input_, param_val, max_iter=100):
    d = input_.shape[0]
    t11 = fw.vec_gen(d)
    z11 = fw.dot(t11, input_)
    t12 = fw.vec_gen(d)
    z12 = fw.dot(t12, input_)
    t13 = fw.vec_gen(d)
    z13 = fw.dot(t13, input_)
    t14 = fw.vec_gen(d)
    z14 = fw.dot(t14, input_)
    t15 = fw.vec_gen(d)
    z15 = fw.dot(t15, input_)

    z1 = fw.concat([z11, z12, z13, z14, z15])
    a1 = fw.ReLu_v(z1)
    # layer 2
    t21 = fw.vec_gen(l1)
    z21 = fw.dot(t21, a1)
    t22 = fw.vec_gen(l1)
    z22 = fw.dot(t22, a1)
    z2 = fw.concat([z21, z22])
    f = fw.sigmoid_v(z2)
    y = np.array([0, 1])
    L = fw.concat([-fw.dot(y, fw.log_v(f)) - fw.dot(1 - y, fw.log_v(1 - f))])

    # params = [t11,t12,t13,t14,t15] + [t21,t22]
    L_val_1 = L.quickeval(param_val)
    L_der_1 = L.quickderiv(param_val)
    print(f.quickeval(param_val))

    L_der = L_der_1
    # print(L.names)
    for i in range(max_iter):
        # for j in range(N):
        param_val -= L_der.flatten() * 0.1 * 1 / (i + 1)
        f_val = f.quickeval(param_val)
        L_val = L.quickeval(param_val)
        L_der = L.quickderiv(param_val)
        # print(f.quickeval(param_val))
        print(i, ' update: ', f_val, ' -> ', L_val)
    # print(L_der_2)
    return param_val


nn(input_, param_val)
