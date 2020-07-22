import numpy as np

"""

m = number of data units (e.g. images)
n = data dimension
l = number of data classes

"""


def compute_softmax(X, W, b):
    """
    m = number of data units (e.g. images)
    n = data dimension
    l = number of data classes
    :param X: data matrix of size n×m
    :param W: weights matrix of size n×l
    :param b: a bias vector of size 1×l
    :return P: probability matrix of size m×l.
                P[i, j] represents the probability of data unit i to be in class j
                by that the sum of each row is supposed to be one
    """
    scores = (X.T @ W) + b  # matrix of size m×l in which each column is X^T ∙ w_j
    exp_scores = np.exp(scores)  # m×l
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)  # m×1
    P = exp_scores / sum_exp_scores  # m×l
    # assert np.linalg.norm(np.sum(P, axis=1) - 1.0) < 1e-10  # remove before submit!!!!!
    return P


def compute_loss(X, y, W, b):
    """

    :param X: data matrix of size n×m
    :param y: labels matrix of size m×l
    :param W: weights matrix of size n×l
    :param b: a bias vector of size 1×l
    :return: loss:
        a scalar representing the entire loss of the data set X, according to softmax function
    """
    m = X.shape[1]
    P = compute_softmax(X, W, b)
    losses = y.T @ np.log(P)
    loss = - (1 / m) * np.sum(np.diag(losses))
    return loss


def compute_loss_gradient_by_W(X, y, W, b):
    """
    :param X: data matrix of size n×m
    :param y: labels matrix of size m×l
    :param W: weights matrix of size n×l
    :param b: a bias vector of size 1×l
    :return: gradient:
        a matrix of size n×l (size of W) representing its gradient
    """
    m = X.shape[1]
    P = compute_softmax(X, W, b)
    gradient = (1 / m) * (X @ (P - y))
    return gradient


def compute_loss_gradient_by_b(X, y, W, b):
    """
    :param X: data matrix of size n×m
    :param y: labels matrix of size m×l
    :param W: weights matrix of size n×l
    :param b: a bias vector of size 1×l
    :return: gradient:
        a vector of size 1×l (size of b) representing its gradient
    """
    m = X.shape[1]
    P = compute_softmax(X, W, b)
    gradient = (1 / m) * np.sum((P - y), axis=0, keepdims=True)
    return gradient


def compute_loss_gradient_by_X(X, y, W, b):
    """
    :param X: data matrix of size n×m
    :param y: labels matrix of size m×l
    :param W: weights matrix of size n×l
    :param b: a bias vector of size 1×l
    :return: gradient:
        a matrix of size n×m (size of X) representing its gradient
    """
    m = X.shape[1]
    WtX = W.T @ X + b.T  # matrix of size l×m in which each row is w_j^T ∙ X
    exp_WtX = np.exp(WtX)  # l×m
    sum_exp_WtX = np.sum(exp_WtX, axis=0, keepdims=True)  # 1×m
    div = exp_WtX / sum_exp_WtX  # l×m
    diff = div - y.T  # l×m
    gradient = (1 / m) * (W @ diff)  # n×m
    return gradient
