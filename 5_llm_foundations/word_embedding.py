# Credit: deeplearning.ai
import re
from collections import defaultdict

import nltk
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def get_idx(words, word2Ind):
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx


def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch = []


def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed


def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C):i] + data[(i + 1):(i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq / num_ctx_words
        yield x, y
        i += 1
        if i >= len(data):
            print('i is being set to 0')
            i = 0


def get_dict(data):
    """
    Input:
        K: the number of negative samples
        data: the data you want to pull from
        indices: a list of word indices
    Output:
        word_dict: a dictionary with the weighted probabilities of each word
        word2Ind: returns dictionary mapping the word to its index
        Ind2Word: returns dictionary mapping the index to its word
    """
    #
    #     words = nltk.word_tokenize(data)
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word


def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = word_tokenize(data)  # tokenize string to words
    data = [ch.lower() for ch in data
            if ch.isalpha()
            or ch == '.'
            ]
    return data


def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i + 1):(i + C + 1)]
        yield context_words, center_word
        i += 1


for x, y in get_windows(['i', 'am', 'happy', 'because', 'i', 'am', 'learning'], 2):
    print(f'{x}\t{y}')


# The CBOW model is based on a neural network.

def relu(z):
    result = z.copy()
    result[result < 0] = 0

    return result


def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)

    return e_z / sum_e_z


def initialize_model(N, V, random_seed=1):
    """
    Inputs:
        N:  dimension of hidden vector
        V:  dimension of vocabulary
        random_seed: random seed for consistent results in the unit tests
     Outputs:
        W1, W2, b1, b2: initialized weights and biases
    """

    np.random.seed(random_seed)

    # W1 has shape (N,V)
    W1 = np.random.rand(N, V)
    # W2 has shape (V,N)
    W2 = np.random.rand(V, N)
    # b1 has shape (N,1)
    b1 = np.random.rand(N, 1)
    # b2 has shape (V,1)
    b2 = np.random.rand(V, 1)
    return W1, W2, b1, b2


def forward_prop(x, W1, W2, b1, b2):
    '''
    Inputs:
        x:  average one hot vector for the context
        W1, W2, b1, b2:  matrices and biases to be learned
     Outputs:
        z:  output score vector
    '''

    # Calculate h
    h = np.dot(W1, x) + b1

    # Apply the relu on h (store result in h)
    h = np.maximum(0, h)

    # Calculate z
    z = np.dot(W2, h) + b2
    return z, h


def compute_cost(y, yhat, batch_size):
    # cost function
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(np.log(1 - yhat), 1 - y)
    cost = - 1 / batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    '''
    Inputs:
        x:  average one hot vector for the context
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  matrices and biases
        batch_size: batch size
     Outputs:
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases
    '''

    # Compute l1 as W2^T (Yhat - Y)
    # Re-use it whenever you see W2^T (Yhat - Y) used to compute a gradient
    l1 = np.dot(W2.T, (yhat - y))
    # Apply relu to l1
    l1 = np.maximum(0, l1)
    # Compute the gradient of W1
    grad_W1 = (1 / batch_size) * np.dot(l1, x.T)  # 1/m * relu(w2.T(yhat-y)) . xT
    # Compute the gradient of W2
    grad_W2 = (1 / batch_size) * np.dot(yhat - y, h.T)
    # Compute the gradient of b1
    grad_b1 = np.sum((1 / batch_size) * np.dot(l1, x.T), axis=1, keepdims=True)
    # Compute the gradient of b2
    grad_b2 = np.sum((1 / batch_size) * np.dot(yhat - y, h.T), axis=1, keepdims=True)

    return grad_W1, grad_W2, grad_b1, grad_b2


def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    '''
    This is the gradient_descent function

      Inputs:
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector
        V:         dimension of vocabulary
        num_iters: number of iterations
     Outputs:
        W1, W2, b1, b2:  updated matrices and biases

    '''
    W1, W2, b1, b2 = initialize_model(N, V, random_seed=42)
    batch_size = 128
    iters = 0
    C = 2
    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        # Get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)
        # Get yhat
        yhat = softmax(z)
        # Get cost
        cost = compute_cost(y, yhat, batch_size)
        if ((iters + 1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        # Get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)

        # Update weights and biases
        W1 -= alpha * grad_W1
        W2 -= alpha * grad_W2
        b1 -= alpha * grad_b1
        b2 -= alpha * grad_b2

        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return W1, W2, b1, b2


sent = 'Learn word embeddings in 2024!!! Why? Becuase it sets the base.'
data = tokenize(sent)
word2Ind, Ind2word = get_dict(data)
print(word2Ind)
print(Ind2word)
num_iters = 150
print("Call gradient_descent")
V = len(word2Ind)
N = 3
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)
print(W1)
print('******************')
print(W2)
print('******************')
embs = (W1.T + W2) / 2.0

for word in data:
    print(word, embs[word2Ind[word]])
