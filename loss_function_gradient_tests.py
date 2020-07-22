from data_utils import get_as_vec
from loss_function import (compute_loss,
                           compute_loss_gradient_by_W,
                           compute_loss_gradient_by_X,
                           compute_loss_gradient_by_b)

import numpy as np
from matplotlib import pyplot as plt

GRADIENT_TESTS_NUM = 2
LOSS_VARS_NUM = 3
EPSILON_ITERATIONS = 19


def plot_gradient_decrease_factors(dec_factors, d_var):
    plt.plot(range(EPSILON_ITERATIONS - 2), dec_factors[:, 0],
             label='|f(%s + ϵd) − f(%s)|' % (d_var, d_var))
    plt.plot(range(EPSILON_ITERATIONS - 2), dec_factors[:, 1],
             label='|f(%s + ϵd) − f(%s) - ϵ*d*grad(%s)|' % (d_var, d_var, d_var))
    plt.xlabel('iterations')
    plt.ylabel('decrease factor')
    plt.title('Loss Function Gradient Test\nDerivative by %s' % d_var)
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


def compute_tests(loss, nxt_loss, grad, epsilon, d):
    grad_test1 = np.abs(nxt_loss - loss)
    d_vec = get_as_vec(d)
    grad_vec = get_as_vec(grad)
    grad_test2 = np.abs(nxt_loss - loss - epsilon * (d_vec.T @ grad_vec))[0, 0]
    return grad_test1, grad_test2


def get_next_step(A, epsilon, d):
    Anxt = np.copy(A)
    Anxt += epsilon * d
    return Anxt


def loss_gradient_tests(m=100, n=30, l=5, num_of_tests=10):
    results = np.zeros((EPSILON_ITERATIONS, LOSS_VARS_NUM * GRADIENT_TESTS_NUM))

    for t in range(num_of_tests):

        X, y, W, b = get_random_params(m, n, l)

        loss = compute_loss(X, y, W, b)

        grad_at_W = compute_loss_gradient_by_W(X, y, W, b)
        grad_by_X = compute_loss_gradient_by_X(X, y, W, b)
        grad_by_b = compute_loss_gradient_by_b(X, y, W, b)

        dW = np.random.rand(*W.shape)
        dX = np.random.rand(*X.shape)
        db = np.random.rand(*b.shape)

        epsilon = 1

        for i in range(EPSILON_ITERATIONS):
            epsilon *= 0.5

            Wnxt = get_next_step(W, epsilon, dW)
            Xnxt = get_next_step(X, epsilon, dX)
            bnxt = get_next_step(b, epsilon, db)

            loss_at_Wnxt = compute_loss(X, y, Wnxt, b)
            loss_at_Xnxt = compute_loss(Xnxt, y, W, b)
            loss_at_bnxt = compute_loss(X, y, W, bnxt)

            W_grad_test1, W_grad_test2 = compute_tests(loss, loss_at_Wnxt, grad_at_W, epsilon, dW)
            X_grad_test1, X_grad_test2 = compute_tests(loss, loss_at_Xnxt, grad_by_X, epsilon, dX)
            b_grad_test1, b_grad_test2 = compute_tests(loss, loss_at_bnxt, grad_by_b, epsilon, db)

            results[i, :] += W_grad_test1, W_grad_test2, X_grad_test1, X_grad_test2, b_grad_test1, b_grad_test2

    results /= num_of_tests
    a = results[2:, ]
    b = results[1:-1, :]
    dec_factors = b / a

    plot_gradient_decrease_factors(dec_factors[:, :2], 'W')
    plot_gradient_decrease_factors(dec_factors[:, 2:4], 'X')
    plot_gradient_decrease_factors(dec_factors[:, 4:], 'b')
