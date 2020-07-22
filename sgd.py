from data_utils import shuffle_data
from loss_function import compute_softmax, compute_loss, compute_loss_gradient_by_W, compute_loss_gradient_by_b
from std_NN.net import Net

import numpy as np
from matplotlib import pyplot as plt

RESULTS_FIELDS = {'Train': 0, 'Test': 1}
FIELDS_NUM = len(RESULTS_FIELDS)
STOP_EPSILON = 1e-5


def plot_results(results, iterations, title=''):
    for field, field_index in RESULTS_FIELDS.items():
        plt.semilogy(range(iterations), results[:, field_index], label=field)
    plt.xlabel('epochs')
    plt.title(title)
    plt.legend()
    plt.show()


def predict(X, W, b):
    P = compute_softmax(X, W, b)
    results = np.argmax(P, axis=1)
    return results


def get_success_rate(X, y, W, b):
    predicted_labels = predict(X, W, b)
    true_labels = np.nonzero(y)[1]
    num_tests = y.shape[0]
    num_successes = np.sum(predicted_labels == true_labels)
    success_rate = num_successes / num_tests
    return success_rate


def get_mini_batches(X_tr, y_tr, mb_num):
    X_tr, y_tr = shuffle_data(X_tr, y_tr)
    mbs = np.array_split(X_tr, mb_num, axis=1)
    mbs_labels = np.array_split(y_tr, mb_num, axis=0)
    return mbs, mbs_labels


def collect_results(X_tr, y_tr, X_te, y_te, W, b, convergence_results, success_rate_results, epoch):
    loss_tr = compute_loss(X_tr, y_tr, W, b)
    loss_te = compute_loss(X_te, y_te, W, b)
    convergence_results[epoch] = loss_tr, loss_te

    success_rate_tr = get_success_rate(X_tr, y_tr, W, b)
    success_rate_te = get_success_rate(X_te, y_te, W, b)
    success_rate_results[epoch] = success_rate_tr, success_rate_te


def sgd(X_tr, y_tr, X_te, y_te, alpha=0.1, mb_num=0, max_epochs=250, data_set=''):
    if not mb_num:
        mb_num = 1

    n, m = X_tr.shape
    l = y_tr.shape[1]
    W = np.random.rand(n, l)
    b = np.random.rand(1, l)

    old_W = np.copy(W)

    convergence_results = np.zeros((max_epochs, FIELDS_NUM))
    success_rate_results = np.zeros((max_epochs, FIELDS_NUM))

    for epoch in range(max_epochs):

        mbs, mbs_labels = get_mini_batches(X_tr, y_tr, mb_num)

        for S, yS in zip(mbs, mbs_labels):
            batch_size = S.shape[1]
            grad_W = (1 / batch_size) * compute_loss_gradient_by_W(S, yS, W, b)
            grad_b = (1 / batch_size) * compute_loss_gradient_by_b(S, yS, W, b)
            W -= alpha * grad_W
            b -= alpha * grad_b

        collect_results(X_tr, y_tr, X_te, y_te, W, b, convergence_results, success_rate_results, epoch)

        if (np.linalg.norm(W - old_W)) < STOP_EPSILON:
            max_epochs = epoch
            break

        old_W = np.copy(W)

    plot_results(convergence_results[:max_epochs], max_epochs, title='%s F(W)' % data_set)
    plot_results(success_rate_results[:max_epochs], max_epochs, title='%s Success Rate' % data_set)

    final_success_rate = success_rate_results[max_epochs-1, RESULTS_FIELDS['Test']]
    print('Dataset = %s, success_rate = %s' % (data_set, final_success_rate))

    return W


def nn_sgd_step(nn, W_grads, b_grads, alpha, batch_size):
    W_steps = [- alpha * (1 / batch_size) * W_grad for W_grad in W_grads]
    b_steps = [- alpha * (1 / batch_size) * b_grad for b_grad in b_grads]
    nn.add_weights_to_layers(W_steps, b_steps)


def nn_sgd(X_tr, y_tr, X_te, y_te, layers=None, alpha=0.1, mb_num=0, max_epochs=250, data_set=''):
    if not mb_num:
        mb_num = 1

    n, m = X_tr.shape
    l = y_tr.shape[1]

    if not layers:
        layers = [n]

    layers_num = len(layers)

    nn = Net(m, l, layers)

    convergence_results = np.zeros((max_epochs, FIELDS_NUM))
    success_rate_results = np.zeros((max_epochs, FIELDS_NUM))

    W, b = nn.get_output_layer_weights()
    old_W = np.copy(W)

    for epoch in range(max_epochs):

        mbs, mbs_labels = get_mini_batches(X_tr, y_tr, mb_num)

        for S, yS in zip(mbs, mbs_labels):
            batch_size = S.shape[1]
            Xs = nn.forward_pass(S)
            W_grads, b_grads = nn.back_prop(Xs, yS)
            nn_sgd_step(nn, W_grads, b_grads, alpha, batch_size)

        Xs_tr = nn.forward_pass(X_tr)
        output_X_tr = Xs_tr[-1]
        Xs_te = nn.forward_pass(X_te)
        output_X_te = Xs_te[-1]
        W, b = nn.get_output_layer_weights()

        collect_results(output_X_tr, y_tr, output_X_te, y_te, W, b, convergence_results, success_rate_results, epoch)

        if (np.linalg.norm(W - old_W)) < STOP_EPSILON:
            max_epochs = epoch
            break

        old_W = np.copy(W)

    plot_results(convergence_results[:max_epochs], max_epochs,
                 title='%s F(W)\n#Layers = %s' % (data_set, layers_num))
    plot_results(success_rate_results[:max_epochs], max_epochs,
                 title='%s Success Rate\n#Layers = %s' % (data_set, layers_num))

    final_success_rate = success_rate_results[max_epochs-1, RESULTS_FIELDS['Test']]
    print('Dataset = %s, Layers = %s, success_rate = %s' % (data_set, layers_num, final_success_rate))

    return W

