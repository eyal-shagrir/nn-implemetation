from data_utils import get_as_vec
from .activation_function import *
import numpy as np
from matplotlib import pyplot as plt

JACOBIAN_TESTS_NUM = 2
ACTIVATION_VARS_NUM = 3
EPSILON_ITERATIONS = 19


def plot_jacobian_decrease_factors(dec_factors, d_var):
    plt.plot(range(EPSILON_ITERATIONS - 2), dec_factors[:, 0],
             label='|f(%s + ϵd) − f(%s)|' % (d_var, d_var))
    plt.plot(range(EPSILON_ITERATIONS - 2), dec_factors[:, 1],
             label='|f(%s + ϵd) − f(%s) - JacMV(%s , ϵd)|' % (d_var, d_var, d_var))
    plt.xlabel('iterations')
    plt.ylabel('decrease factor')
    plt.title('Activation Function Jacobian Test\nDerivative by %s' % d_var)
    plt.legend()
    plt.show()


def get_random_params(m, n, k):
    X = np.random.rand(n, m)
    W = np.random.uniform(low=-1, high=1, size=(k, n))
    b = np.random.uniform(low=-5, high=5, size=(k, 1))
    return X, W, b


def compute_tests(loss, nxt_loss, JWV):
    grad_test1 = np.linalg.norm(nxt_loss - loss)
    grad_test2 = np.linalg.norm(nxt_loss - loss - JWV)
    return grad_test1, grad_test2


def get_next_step(A, epsilon, d):
    Anxt = np.copy(A)
    Anxt += epsilon * d
    return Anxt


def activation_jacobian_tests(m=100, n=30, k=5, num_of_tests=10):
    results = np.zeros((EPSILON_ITERATIONS, ACTIVATION_VARS_NUM * JACOBIAN_TESTS_NUM))

    for t in range(num_of_tests):

        X, W, b = get_random_params(m, n, k)

        f = compute_tanh(X, W, b)

        dW = np.random.rand(*W.shape)
        dX = np.random.rand(*X.shape)
        db = np.random.rand(*b.shape)

        epsilon = 1

        for i in range(EPSILON_ITERATIONS):
            epsilon *= 0.5

            Wnxt = get_next_step(W, epsilon, dW)
            Xnxt = get_next_step(X, epsilon, dX)
            bnxt = get_next_step(b, epsilon, db)

            f_at_Wnxt = compute_tanh(X, Wnxt, b)
            f_at_Xnxt = compute_tanh(Xnxt, W, b)
            f_at_bnxt = compute_tanh(X, W, bnxt)

            JWV = compute_tanh_JWV(X, W, b, epsilon * dW)
            JXV = compute_tanh_JXV(X, W, b, epsilon * dX)
            JbV = compute_tanh_JbV(X, W, b, epsilon * db)

            W_JMV_test1, W_JMV_test2 = compute_tests(f, f_at_Wnxt, JWV)
            X_JMV_test1, X_JMV_test2 = compute_tests(f, f_at_Xnxt, JXV)
            b_JMV_test1, b_JMV_test2 = compute_tests(f, f_at_bnxt, JbV)

            results[i, :] += W_JMV_test1, W_JMV_test2, X_JMV_test1, X_JMV_test2, b_JMV_test1, b_JMV_test2

    results /= num_of_tests
    a = results[2:, ]
    b = results[1:-1, :]
    dec_factors = b / a

    plot_jacobian_decrease_factors(dec_factors[:, :2], 'W')
    plot_jacobian_decrease_factors(dec_factors[:, 2:4], 'X')
    plot_jacobian_decrease_factors(dec_factors[:, 4:], 'b')


def activation_transpose_jacobian_tests(m=2, n=3, k=5, num_of_tests=10):
    X, W, b = get_random_params(m, n, k)

    for t in range(num_of_tests):
        V = np.random.rand(k, n)  # k×n
        U = np.random.rand(k, m)  # k×m
        V_vec = get_as_vec(V)  # kn×1
        U_vec = get_as_vec(U)  # km×1
        JWV = compute_tanh_JWV(X, W, b, V)  # k×m
        JWTV = compute_tanh_JWTV(X, W, b, U)  # k×n
        JWV_vec = get_as_vec(JWV)  # km×1
        JWTV_vec = get_as_vec(JWTV)  # kn×1
        diff = U_vec.T @ JWV_vec - V_vec.T @ JWTV_vec
        assert np.linalg.norm(diff) < 1e-10
    print('W transpose jacobian test - pass!')

    for t in range(num_of_tests):
        V = np.random.rand(n, m)  # n×m
        U = np.random.rand(k, m)  # k×m
        V_vec = get_as_vec(V)  # nm×1
        U_vec = get_as_vec(U)  # km×1
        JXV = compute_tanh_JXV(X, W, b, V)  # k×m
        JXTV = compute_tanh_JXTV(X, W, b, U)  # n×m
        JXV_vec = get_as_vec(JXV)  # km×1
        JXTV_vec = get_as_vec(JXTV)  # nm×1
        diff = U_vec.T @ JXV_vec - V_vec.T @ JXTV_vec
        assert np.linalg.norm(diff) < 1e-10
    print('X transpose jacobian test - pass!')

    for t in range(num_of_tests):
        V = np.random.rand(k, 1)  # k×1
        U = np.random.rand(k, m)  # k×m
        V_vec = get_as_vec(V)  # k×1
        U_vec = get_as_vec(U)  # km×1
        JbV = compute_tanh_JbV(X, W, b, V)  # k×m
        JbTV = compute_tanh_JbTV(X, W, b, U)  # k×1
        JbV_vec = get_as_vec(JbV)  # km×1
        JbTV_vec = get_as_vec(JbTV)  # k×1
        diff = U_vec.T @ JbV_vec - V_vec.T @ JbTV_vec
        assert np.linalg.norm(diff) < 1e-10

    print('b transpose jacobian test - pass!')
