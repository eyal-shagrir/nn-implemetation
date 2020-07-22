import numpy as np


def compute_tanh(X, W, b):
    """
    :param X: data matrix of size n×m
    :param W: weights matrix of size k×n
    :param b: bias vector of size k×1
    :return: matrix of size k×m
    """
    scores = W @ X + b  # k×m
    tanh_scores = np.tanh(scores)  # k×m
    return tanh_scores


def compute_tanh_derivative(X, W, b):
    tanh = compute_tanh(X, W, b)  # k×m
    tanh_by_2 = np.power(tanh, 2)  # k×m
    tanh_derivative = 1 - tanh_by_2  # k×m
    return tanh_derivative


def compute_tanh_JWV(X, W, b, V):
    """
    :param X: data matrix of size n×m
    :param W: weights matrix of size k×n
    :param b: bias vector of size k×1
    :param V: a matrix of size k×n
    :return: JWV: matrix of size k×m representing J_W ∙ V
    """
    D = compute_tanh_derivative(X, W, b)  # k×m
    VX = V @ X  # k×m
    JWV = D * VX
    return JWV


def compute_tanh_JWTV(X, W, b, V):
    """
    :param X: data matrix of size n×m
    :param W: weights matrix of size k×n
    :param b: bias vector of size k×1
    :param V: a matrix of size k×m
    :return: JWTV: matrix of size k×n representing J_W^T ∙ V
    """
    D = compute_tanh_derivative(X, W, b)  # k×m
    DV = D * V  # k×m
    JWTV = DV @ X.T  # k×n
    return JWTV


def compute_tanh_JXV(X, W, b, V):
    """
    :param X: data matrix of size n×m
    :param W: weights matrix of size k×n
    :param b: bias vector of size k×1
    :param V: a matrix of size n×m
    :return: JXV: matrix of size k×m representing J_X ∙ V
    """
    D = compute_tanh_derivative(X, W, b)  # k×m
    WV = W @ V   # k×m
    JXV = D * WV  # k×m
    return JXV


def compute_tanh_JXTV(X, W, b, V):
    """
    :param X: data matrix of size n×m
    :param W: weights matrix of size k×n
    :param b: bias vector of size k×1
    :param V: a matrix of size k×m
    :return: JXTV: matrix of size n×m representing J_X.T ∙ V
    """
    D = compute_tanh_derivative(X, W, b)  # k×m
    DV = D * V  # k×m
    JXTV = W.T @ DV  # n×m
    return JXTV


def compute_tanh_JbV(X, W, b, V):
    """
    :param X: data matrix of size n×m
    :param W: weights matrix of size k×n
    :param b: bias vector of size k×1
    :param V: a matrix of size k×1
    :return: JbV: vector of size k×m representing J_b ∙ V
    """
    D = compute_tanh_derivative(X, W, b)  # k×m
    JbV = D * V  # k×m
    return JbV


def compute_tanh_JbTV(X, W, b, V):
    """
    :param X: data matrix of size n×m
    :param W: weights matrix of size k×n
    :param b: bias vector of size k×1
    :param V: a matrix of size k×m
    :return: JbTV: matrix of size k×1 representing J_b.T ∙ V
    """
    D = compute_tanh_derivative(X, W, b)  # k×m
    DV = D * V  # k×m
    JbTV = np.sum(DV, axis=1, keepdims=True)  # k×1
    return JbTV
