import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, Embedding, Lambda
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Expanded sentence pairs with synonyms
sentence_pairs = np.array([
    ["What is artificial intelligence?", "Explain AI", 1],
    ["How to learn Python?", "Best way to study Python", 1],
    ["Capital of France", "Paris is the capital of which country?", 1],
    ["What is your age?", "How old are you?", 1],
    ["Describe machine learning", "Explain ML", 1],
    ["Weather in New York", "New York weather forecast", 1],
    ["Solve math problems", "How to solve mathematics", 1],
    ["Read a book", "How to engage in reading", 1],
    ["Running benefits", "Advantages of jogging", 1],
    ["Healthy diet plan", "Plan for a healthy diet", 1],
    ["What is the capital of Germany?", "Who is the president of the USA?", 0],
    ["Learn to swim", "Learn to dance", 0],
    ["Cooking a cake", "Baking a pie", 0],
    ["Play football", "Play guitar", 0],
    ["Ocean animals", "Mountain wildlife", 0],
    ["Speak English", "Speak Spanish", 0],
    ["Travel by plane", "Travel by train", 0],
    ["Go fishing", "Go hunting", 0],
    ["Watching movies", "Listening to music", 0],
    ["Painting a picture", "Writing a poem", 0]
    ["Where are you going?", "Where are you from?", 0]
    ["Where do you reside?", "Where are you from?", 1]
])

# Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence_pairs[:, 0].tolist() + sentence_pairs[:, 1].tolist())
max_length = max([len(x.split()) for x in np.concatenate([sentence_pairs[:, 0], sentence_pairs[:, 1]])])

q1_sequences = tokenizer.texts_to_sequences(sentence_pairs[:, 0])
q2_sequences = tokenizer.texts_to_sequences(sentence_pairs[:, 1])

q1_data = pad_sequences(q1_sequences, maxlen=max_length)
q2_data = pad_sequences(q2_sequences, maxlen=max_length)
labels = sentence_pairs[:, 2].astype('float32')


# Siamese LSTM Model with Cosine Similarity
def create_siamese_lstm_model():
    embedding_dim = 50

    input_a = Input(shape=(max_length,))
    input_b = Input(shape=(max_length,))

    embedding_layer = Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_length)
    shared_lstm = LSTM(64)

    encoded_a = shared_lstm(embedding_layer(input_a))
    encoded_b = shared_lstm(embedding_layer(input_b))

    # Cosine similarity function
    def cosine_similarity(vects):
        x, y = vects
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return tf.reduce_sum(tf.multiply(x, y), axis=-1, keepdims=True)

    similarity = Lambda(cosine_similarity)([encoded_a, encoded_b])

    model = Model(inputs=[input_a, input_b], outputs=similarity)

    return model


model = create_siamese_lstm_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([q1_data, q2_data], labels, epochs=20, batch_size=16)  # Increased epochs for better learning


# Function to predict similarity of new sentences
def predict_similarity(sentence1, sentence2):
    sequence1 = tokenizer.texts_to_sequences([sentence1])
    sequence2 = tokenizer.texts_to_sequences([sentence2])

    data1 = pad_sequences(sequence1, maxlen=max_length)
    data2 = pad_sequences(sequence2, maxlen=max_length)

    prediction = model.predict([data1, data2])
    return prediction[0][0]


# Example prediction

similarity_score = predict_similarity("How to stay fit?", "What are some ways to maintain fitness?")
print(f"Similarity score: {similarity_score}")
similarity_score = predict_similarity("I want to go to gym", "Office is over")
print(f"Similarity score: {similarity_score}")
similarity_score = predict_similarity("How young are you?", "What is your age?")
print(f"Similarity score: {similarity_score}")
