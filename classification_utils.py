from loss_function import compute_softmax
import numpy as np

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
