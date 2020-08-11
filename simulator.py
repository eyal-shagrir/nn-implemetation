from data_utils import get_data
from loss_function_gradient_tests import loss_gradient_tests
from sgd import sgd, nn_sgd
from std_NN.activation_jacobian_tests import activation_jacobian_tests, activation_transpose_jacobian_tests
from std_NN.nn_gradient_tests import nn_gradient_tests


def task1():
    loss_gradient_tests()


def task3():
    sgd_params = {'GMM': {'alpha': 1.0, 'mb_num': 200},
                  'Peaks': {'alpha': 0.1, 'mb_num': 250},
                  'SwissRoll': {'alpha': 0.08, 'mb_num': 250}}

    for data_set, params in sgd_params.items():
        X_tr, y_tr, X_te, y_te = get_data(data_set)
        _ = sgd(X_tr, y_tr, X_te, y_te,
                alpha=params['alpha'],
                mb_num=params['mb_num'],
                max_epochs=200,
                data_set=data_set)


def task4():
    activation_jacobian_tests()
    activation_transpose_jacobian_tests()


def task6():
    nn_gradient_tests()


def task7():
    nn_sgd_params = {'GMM': {'layers_nums': (1, 3, 5, 10, 15), 'alpha': 1.0, 'mb_num': 200},
                     'Peaks': {'layers_nums': (1, 3, 5, 10, 15), 'alpha': 1.0, 'mb_num': 200},
                     'SwissRoll': {'layers_nums': (1, 3, 5, 10), 'alpha': 1.0, 'mb_num': 200}}

    for data_set, params in nn_sgd_params.items():
        X_tr, y_tr, X_te, y_te = get_data(data_set)
        n = X_tr.shape[0]
        layers_nums = params['layers_nums']
        for layers_num in layers_nums:
            layers = [n + 5 * i for i in range(1, layers_num)]
            _ = nn_sgd(X_tr, y_tr, X_te, y_te,
                       hidden_layers_sizes=layers,
                       alpha=params['alpha'],
                       mb_num=params['mb_num'],
                       max_epochs=200,
                       data_set=data_set)


def main():
    # task1()  # loss function gradient tests
    # task3()  # minimizing loss function by SGD
    # task4()  # activation function jacobian tests
    task6()  # NN gradient tests
    task7()  # minimizing loss function by SGD with NN


if __name__ == '__main__':
    main()
