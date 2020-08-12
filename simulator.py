from data_utils import get_data
from loss_function_gradient_tests import loss_gradient_tests
from std_NN.activation_jacobian_tests import activation_jacobian_tests, activation_transpose_jacobian_tests
from std_NN.nn_gradient_tests import nn_gradient_tests

from std_NN.net import Net


def run_derivative_tests():
    loss_gradient_tests()
    activation_jacobian_tests()
    activation_transpose_jacobian_tests()
    nn_gradient_tests()


def main():
    # run_derivative_tests()

    X_tr, y_tr, X_te, y_te = get_data('GMM')
    n, m = X_tr.shape
    c = y_tr.shape[1]
    nn = Net(n, c, [n+5, n+10])
    nn.train(X_tr, y_tr, alpha=1.0, mb_num=200, max_epochs=10, data_set='GMM', plot=True)
    sr = nn.predict(X_te, y_te)
    print(sr)


if __name__ == '__main__':
    main()
