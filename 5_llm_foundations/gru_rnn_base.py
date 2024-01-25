import numpy as np
import tensorflow as tf

"""
h_t = g(w_h[h_t-1, x_t] + b_h)
h_t = g(w_hh.h_t-1 + w_hx.x_t + b_h)
y_t = g(W_yh.h_t + b_y)
J = (1/T) * summation(y_t.log(y_hat_t))

Sequential A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
Dense A regular fully connected layer
GRU The GRU (gated recurrent unit) layer. The hidden state dimension should be specified (the syntax is the same as for Dense). 
By default it does not return a sequence, but only the output of the last unit. If you want to stack two consecutive GRU layers, you need the first one to output a sequence, which you can achieve by setting the parameter return_sequences to True. If you are further interested in similar layers, you can also check out the RNN, LSTM and Bidirectional. If you want to use a RNN or LSTM instead of GRU in the code below, simply change the layer name, no other change in the syntax is needed.

"""

model_GRU = tf.keras.Sequential([
    tf.keras.layers.GRU(256, return_sequences=True, name='GRU_1_returns_seq'),
    tf.keras.layers.GRU(128, return_sequences=True, name='GRU_2_returns_seq'),
    tf.keras.layers.GRU(64, name='GRU_3_returns_last_only'),
    tf.keras.layers.Dense(10)
])

batch_size = 60
sequence_length = 50
word_vector_length = 40

input_data = tf.random.normal([batch_size, sequence_length, word_vector_length])

# Pass the data through the network
prediction = model_GRU(input_data)

# Show the summary of the model
print(model_GRU.summary())

