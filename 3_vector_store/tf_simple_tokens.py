from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Define a small corpus of sentences
corpus = [
    "hello world",
    "hello from the other side",
    "deep learning is fun",
    "artificial intelligence is the future",
    "hello artificial world",
    "This is an artificial world",
    "Fake & artificial world",
    "hello artificial intelligence",
]

# Step 2: Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(tokenizer.word_index)

# Convert sentences to sequences of integers
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to the same length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
print(input_sequences)
