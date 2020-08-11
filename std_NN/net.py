from loss_function import *
from .activation_function import *
import numpy as np

"""
n = the dimension of the original data
m =  number of data examples
c =  number of classes
l = number of hidden layers in the net

L is a layer.
The network is assembled from the following layers (L_0, L_1, L_2, ..., L_l, L*),
where L_0 is the input layer, and L* is the output layer.

Accordingly, we define the number of neurons in each layer by (n_0, n_1, ..., n_l, n*), i.e L_i contains n_i neurons.
Thus, n_0=n, and n*=c.

The neurons of each layer that gets data (except L*), activate an activation function on it and passes forward.

In practice, each layer L_i∈(L_1, L_2, .., L_l, L*) is the weights (W_i, b_i) of the connections between the neurons of
L_(i-1) and L_i.
For example, L_1 is the weights (W_1, b_1) of the connections between the neurons of L_0 and neurons of L_1.
In the same way, L* is the weights (W*, b*) of the connections between the neurons of L_l and neurons of L*.

We also define (X_0, X_1, ..., X_l, X*), where X_i is the data outputs from layer L_i, i.e. after it passes the 
through the activation function.
Therefore, X_0 is the original data, and X* is the scores of the classes. 

As for dimensions:
1. The input layer L_0:
    - Contains n_0=n neurons
    - Has no weights.
    - It outputs the original data X_0∈n×m
2. Hidden layer L_i:
    - Contains n_i neurons
    - Has weights (W_i, b_i) where W_i∈n_i×n_(i-1), b∈n_i×1
    - Gets as input X_(i-1)∈n_(i-1)×m
    - Outputs X_i=ϕ(W_i∙X_(i-1) +b_i), where X_i∈n_i×m
3. Output layer L*:
    - Contains c neurons
    - Has weights (W*, b*) where W*∈n_l×c, b∈1×c
    - Gets as input X_l∈n_l×m
    - Outputs X*=softmax((X_l)^T∙W* + b*), where X*∈m×c    

"""


class Net:

    def __init__(self, n, m, c, hidden_layers_sizes):
        """
        creates a standard neural network with len(ns) layers
        :param n: the dimension of the original data, and also number of neurons in the input layer
        :param m: number of data examples
        :param c: number of classes, and also number of neurons in the output layer
        :param hidden_layers_sizes:
            a list of the hidden layers' sizes, i.e. the number of neurons in each hidden layer: (n_1, n_2, ..., n_l)
        """
        self.n = n
        self.m = m
        self.c = c

        self.hidden_layers_sizes = hidden_layers_sizes
        self.hidden_layers_num = len(hidden_layers_sizes)

        self.layers = self.create_layers()
        self.hidden_layers = self.layers[:-1]

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
        initializes the proper weights (W, b) for each layer in the net
        """
        layers = []
        n = self.n
        c = self.c
        for k in self.hidden_layers_sizes:
            hidden_layer = self.create_layer(k, n, k, 1)
            layers.append(hidden_layer)
            n = k
        output_layer = self.create_layer(n, c, 1, c)
        layers.append(output_layer)
        return layers

    def forward_pass(self, X):
        """
        computes the data manipulation at each layer by the activation function
        :param: X: the original data matrix
        :return: Xs: the neurons of each layer (X_0, X_1, ...) where X_i is the neurons of layer i
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


