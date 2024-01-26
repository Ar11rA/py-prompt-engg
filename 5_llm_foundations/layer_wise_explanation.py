import numpy as np

from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# Define a simple corpus
corpus = [
    'the quick brown fox jumps',
    'over the lazy dog and',
    'the quick brown dog jumps',
    'over the lazy fox and',
]

# Tokenize the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
# Pad sequences
max_sequence_len = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
labels = to_categorical(labels, num_classes=vocab_size)

# Create the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=3, input_length=max_sequence_len - 1))
model.add(SimpleRNN(units=4))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model (Note: In practice, you'd use more epochs to train the model)
model.fit(predictors, labels, epochs=20, verbose=1)

# To print the outputs of each layer, we use a Keras backend function
from tensorflow.keras import backend as K


# Define a function to print the output of each layer
def get_layer_output(model, input_data, layer_number):
    get_output = K.function([model.layers[0].input],
                            [model.layers[layer_number].output])
    return get_output([input_data])[0]


# Print outputs of each layer for the first sequence

for i in range(len(model.layers)):
    layer_output = get_layer_output(model, predictors[0:4], i)
    print(f'Layer {i} output:')
    print(layer_output)

print(tokenizer.word_index)

new_sentence = "the quick brown fox"

# Preprocess the new sentence
token_list = tokenizer.texts_to_sequences([new_sentence])[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

# Predict the next word
predicted_probs = model.predict(token_list)
predicted_index = np.argmax(predicted_probs, axis=-1)[0]

# Retrieve the word from the tokenizer's index_word
predicted_word = tokenizer.index_word[predicted_index]

print("Predicted last word:", predicted_word)