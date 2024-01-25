import numpy as np
from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

corpus = [
    "John and Mary went to the market",
    "Emily and Kevin saw a movie last night",
    "The weather in New York is quite unpredictable",
    "David and Samantha are planning a trip to Paris",
    "The new song by Elton John is amazing",
    "California has some beautiful beaches",
    "Dr. Watson and Sherlock Holmes solved the mystery",
    "The book on the table belongs to Natalie",
    "Mark and Jacob are brothers",
    "Julia is studying computer science at Stanford",
    "The tallest building in Dubai is quite a sight",
    "I will meet Alice and Bob at the conference",
    "The lecture by Professor Johnson was very enlightening",
    "Olivia is a professional photographer",
    "The paintings by Vincent Van Gogh are priceless",
    "Friend of John is Mark",
    "Friends are everywhere of John",
    "John and Mark are good"
]

tags = [
    "P O P O O O O",
    "P O P O O O O O",
    "O O O P P O O O",
    "P O P O O O O O P",
    "O O O O P P O O",
    "P O O O O O",
    "O P O P P O O O",
    "O O O O O O O P",
    "P O P O O",
    "P O O O O O O P",
    "O O O O P O O O",
    "O O O P O P O O O",
    "O O O O P O O O O",
    "P O O O O",
    "O O O P P P O O",
    "O O P O P",
    "O O O O P",
    "P O P O O"
]


# Preprocessing the corpus
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(corpus)
vocab_size = len(word_tokenizer.word_index) + 1  # Plus 1 for padding
max_len = max([len(sentence.split()) for sentence in corpus])

# Convert sentences to sequences
corpus_seq = word_tokenizer.texts_to_sequences(corpus)
corpus_padded = pad_sequences(corpus_seq, maxlen=max_len, padding='post')

# Preprocessing the tags
tag_map = {'O': 0, 'P': 1}
tag_tokenizer = Tokenizer(num_words=len(tag_map) + 1, filters='', oov_token='[UNK]')
tag_tokenizer.fit_on_texts(tags)

# Convert tags to sequences
tags_seq = tag_tokenizer.texts_to_sequences(tags)
tags_padded = pad_sequences(tags_seq, maxlen=max_len, padding='post')

print(len(tag_map))
print(tags_padded)

# Define the LSTM Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dense(len(tag_map) + 1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(corpus_padded, tags_padded, batch_size=1, epochs=10)

# Example of prediction
test_sentence = ["John and Emily are friends"]
test_seq = word_tokenizer.texts_to_sequences(test_sentence)
test_padded = pad_sequences(test_seq, maxlen=max_len, padding='post')
prediction = model.predict(test_padded)
print(test_padded)
prediction_tags = np.argmax(prediction, axis=-1)
print(prediction)
print(prediction_tags)
