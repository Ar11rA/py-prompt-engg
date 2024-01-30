import numpy as np

# Assuming we have the following query, keys, and values:
query = np.array([[1, 0.9]])  # For "c'est"
keys = np.array([
    [1, 0.8],  # Key for "it's"
    [0.9, 1],  # Key for "time"
    [1, 0.9],  # Key for "for"
    [0.9, 0.8]  # Key for "tea"
])
values = np.array([
    [1, 2],  # Value for "it's"
    [2, 3],  # Value for "time"
    [3, 4],  # Value for "for"
    [4, 5]  # Value for "tea"
])

# Calculate the dot products of the query with all keys
dot_products = np.dot(query, keys.T)

# Scale the dot products
d_k = query.shape[1]  # Dimension of the keys and query
scaled_dot_products = dot_products / np.sqrt(d_k)

# Apply softmax to get the weights
attention_weights = np.exp(scaled_dot_products) / np.sum(np.exp(scaled_dot_products), axis=-1, keepdims=True)

# Multiply the weights by the values to get the attention output
attention_output = np.dot(attention_weights, values)

print(attention_output, attention_weights)
