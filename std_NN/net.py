from loss_function import *
from .activation_function import *
import numpy as np


class Net:

    def __init__(self, m, l, ns):
        """
        creates a standard neural network with len(ns) layers
        :param m: number of data units
        :param l: number of classes
        :param ns:
            a list of sizes (n_1, n_2, ...)
            n_i is the dimension of data inputs to layer i
            i.e. the data matrix X inputs to layer i, is of size n_i×m
        """
        self.m = m
        self.l = l

        self.layers_dims = ns
        self.original_n = ns[0]
        self.layers = self.create_layers()
        self.layers_num = len(self.layers)

        self.hidden_layers_dims = ns[:-1]
        self.hidden_layers = self.layers[:-1]
        self.hidden_layers_num = len(self.hidden_layers_dims)

        self.Xs = None

    def create_layer(self, w1, w2, b1, b2):
        """
        creates a layer according to dims given
        the weights of the layer (W, b) are random
        """
        W = np.random.uniform(low=-1, high=1, size=(w1, w2))
        b = np.random.uniform(low=-1, high=1, size=(b1, b2))
        layer = Layer(W, b)
        return layer

    def create_layers(self):
        """
        creates the neural network layers
        the dims are as follows:
            if L_i is an hidden layer, i.e not the output layer:
                X∈n_i×m, W∈k×n_i, b∈k×1, where k=n_(i+1) (k is the dim the data "grows" into)
            if L_i is an output layer:
                X∈n*, W∈n*×l, b∈1×l, where n* is the last n in ns (n* is the final dim of the data)
        """
        layers = []
        n = self.original_n
        l = self.l
        for k in self.layers_dims[1:]:
            hidden_layer = self.create_layer(k, n, k, 1)
            layers.append(hidden_layer)
            n = k
        output_layer = self.create_layer(n, l, 1, l)
        layers.append(output_layer)
        return layers

    def forward_pass(self, X):
        """
        computes the data manipulation at each layer by the activation function
        :param: X: the original data matrix
        :return: Xs: a list of data matrices (X_1, X_2, ...) where X_i is data inputs to layer i
        """
        Xs = [X]
        for layer in self.hidden_layers:
            W, b = layer.W, layer.b
            new_X = compute_tanh(X, W, b)
            X = new_X
            Xs += [X]
        return Xs

    def back_prop(self, Xs, y):
        """
        computes the net gradients with respect ro each layer's weights (W, b)
        :return: W_grads, b_grads:
                    two lists of the gradients: (Wg_1, Wg_2, ...), (bg_1, bg_2, ...)
                    where Wg_i and bg_i are the gradients of the weights of layer i
        """
        W_grads, b_grads = [], []
        final_X = Xs[-1]
        final_W, final_b = self.get_output_layer_weights()
        loss_grad_by_W = compute_loss_gradient_by_W(final_X, y, final_W, final_b)
        loss_grad_by_b = compute_loss_gradient_by_b(final_X, y, final_W, final_b)

        W_grads += [loss_grad_by_W]
        b_grads += [loss_grad_by_b]

        if len(Xs) > 1:  # if there are hidden layers
            loss_grad_by_X = compute_loss_gradient_by_X(final_X, y, final_W, final_b)
            reversed_Xs = list(reversed(Xs))
            i = 1
            tail = loss_grad_by_X
            for hidden_layer in reversed(self.hidden_layers):
                W, b = hidden_layer.W, hidden_layer.b
                X = reversed_Xs[i]
                JWTV = compute_tanh_JWTV(X, W, b, tail)
                JbTV = compute_tanh_JbTV(X, W, b, tail)
                W_grads += [JWTV]
                b_grads += [JbTV]
                JXTV = compute_tanh_JXTV(X, W, b, tail)
                tail = JXTV
                i += 1

        return list(reversed(W_grads)), list(reversed(b_grads))

    def get_layers_weights(self):
        Ws, bs = [], []
        for layer in self.layers:
            W, b = layer.get_layer_weights()
            Ws += [W]
            bs += [b]
        return Ws, bs

    def get_output_layer_weights(self):
        output_layer = self.layers[-1]
        W, b = output_layer.get_layer_weights()
        return W, b

    def set_layers_weights(self, Ws, bs):
        # assert len(Ws) == len(bs) == self.layers_num  # REMOVE BEFORE SUBMIT!
        i = 0
        for W, b in zip(Ws, bs):
            layer = self.layers[i]
            layer.W = W
            layer.b = b
            i += 1

    def add_weights_to_layers(self, dWs, dbs):
        i = 0
        for dW, db in zip(dWs, dbs):
            layer = self.layers[i]
            layer.W += dW
            layer.b += db
            i += 1


class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def get_layer_weights(self):
        return self.W, self.b
