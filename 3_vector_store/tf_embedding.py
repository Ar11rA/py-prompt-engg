import tensorflow as tf
import numpy as np

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


tf.random.set_seed(42)

# Define your corpus
corpus = [
    "hello world",
    "hello artificial intelligence",
    "hello from the other side",
    "deep learning is fun",
    "artificial intelligence is the future",
    "hello artificial world",
    "This is an artificial world",
    "Fake & artificial world",
]

# Tokenize the corpus using TensorFlow's Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

# Vocabulary and word-to-index mapping
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # Add 1 for the padding token

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Define the embedding dimension
embedding_dim = 10

# Create an embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)

# Generate word embeddings using the embedding layer
word_embeddings = embedding_layer(padded_sequences)
print(word_embeddings.shape)
print(word_embeddings[0][0])

model = tf.keras.models.Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

# Get the embeddings
embeddings = model.predict(padded_sequences)
print(embeddings.shape)
print(embeddings[0][0])

print(np.allclose(word_embeddings[0][0], embeddings[0][0]))
