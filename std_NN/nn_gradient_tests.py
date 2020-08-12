from data_utils import get_as_vec
from loss_function import compute_loss
from .net import Net

import numpy as np
from matplotlib import pyplot as plt

GRADIENT_TESTS_NUM = 2
LOSS_VARS_NUM = 2
EPSILON_ITERATIONS = 19


def plot_gradient_decrease_factors(dec_factors, d_var):
    plt.plot(range(EPSILON_ITERATIONS - 2), dec_factors[:, 0],
             label='|f(%s + ϵd) − f(%s)|' % (d_var, d_var))
    plt.plot(range(EPSILON_ITERATIONS - 2), dec_factors[:, 1],
             label='|f(%s + ϵd) − f(%s) - ϵ*d*grad(%s)|' % (d_var, d_var, d_var))
    plt.xlabel('iterations')
    plt.ylabel('decrease factor')
    plt.title('Entire NN Gradient Test\nDerivative by %s' % d_var)
    plt.legend()
    plt.show()


def get_random_params(m, n, l):
    X = np.random.rand(n, m)
    W = np.random.uniform(low=-1, high=1, size=(n, l))
    b = np.random.uniform(low=-5, high=5, size=(1, l))

    y = np.zeros((m, l))
    y_indices = np.random.randint(l, size=(m,))
    y[np.arange(m), y_indices] = 1

    return X, y, W, b


def compute_tests(loss, nxt_loss, grads, epsilon, ds):
    grad_test1 = np.abs(nxt_loss - loss)
    d_vec = np.vstack([get_as_vec(d) for d in ds])
    grad_vec = np.vstack([get_as_vec(g) for g in grads])
    grad_test2 = np.abs(nxt_loss - loss - epsilon * (d_vec.T @ grad_vec))[0, 0]
    return grad_test1, grad_test2


def get_next_steps(As, epsilon, ds):
    nxts = []
    for A, d in zip(As, ds):
        Anxt = A + epsilon * d
        nxts += [Anxt]
    return nxts


def get_next_f(nn, X, y, Ws, bs):
    nn.set_layers_weights(Ws, bs)
    Xs = nn.forward_pass(X)
    W, b = nn.get_output_layer_weights()
    nxt_f = compute_loss(Xs[-1], y, W, b)
    return nxt_f


def nn_gradient_tests(m=100, n=30, l=5, num_of_tests=10):
    results = np.zeros((EPSILON_ITERATIONS, LOSS_VARS_NUM * GRADIENT_TESTS_NUM))

    for t in range(num_of_tests):

        X, y, _, _ = get_random_params(m, n, l)

        nn = Net(n, l, [n+1, n+2])

        Xs = nn.forward_pass(X)
        W, b = nn.get_output_layer_weights()
        f = compute_loss(Xs[-1], y, W, b)

        Ws, bs = nn.get_layers_weights()
        W_grads, b_grads = nn.back_prop(Xs, y)

        dWs = [np.random.rand(*W.shape) for W in Ws]
        dbs = [np.random.rand(*b.shape) for b in bs]

        epsilon = 1

        for i in range(EPSILON_ITERATIONS):
            epsilon *= 0.5

            nxt_Ws = get_next_steps(Ws, epsilon, dWs)
            nxt_bs = get_next_steps(bs, epsilon, dbs)

            f_at_Wnxts = get_next_f(nn, X, y, nxt_Ws, bs)
            f_at_bnxts = get_next_f(nn, X, y, Ws, nxt_bs)

            W_grad_test1, W_grad_test2 = compute_tests(f, f_at_Wnxts, W_grads, epsilon, dWs)
            b_grad_test1, b_grad_test2 = compute_tests(f, f_at_bnxts, b_grads, epsilon, dbs)

            results[i, :] += W_grad_test1, W_grad_test2, b_grad_test1, b_grad_test2

    results /= num_of_tests
    a = results[2:, ]
    b = results[1:-1, :]
    dec_factors = b / a

    plot_gradient_decrease_factors(dec_factors[:, :2], 'W')
    plot_gradient_decrease_factors(dec_factors[:, 2:], 'b')
