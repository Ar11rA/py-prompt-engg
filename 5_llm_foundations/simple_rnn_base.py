import numpy as np

from keras.layers import Embedding, SimpleRNN, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

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

# Predictors and label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
print(predictors, label)
label = to_categorical(label, num_classes=total_words)

# Step 3: Build the RNN model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))
model.add(SimpleRNN(100))  # replace with GRU or LSTM
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model (this is a simple example, so the number of epochs is small)
model.fit(predictors, label, epochs=100, verbose=1)


# Function to predict the next word
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=-1)[0]
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break
    return output_word


# Test the model with some seeds
seed_texts = [
    "hello from",
    "deep learning",
    "hello artificial"
]
predicted_words = [predict_next_word(seed) for seed in seed_texts]

print(predicted_words)
