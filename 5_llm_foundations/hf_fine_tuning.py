import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertForSequenceClassification

# Sample corpus with labels (1: positive, 0: negative)
corpus = [
    ("I love this product", 1),
    ("This is an excellent movie", 1),
    ("I hate this", 0),
    ("This is terrible", 0),
    ("This is terrible movie", 0),
    ("Such a garbage movie", 0)
]

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the corpus
train_encodings = tokenizer([sentence for sentence, label in corpus], truncation=True, padding=True, max_length=128)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    [label for sentence, label in corpus]
))

# Load the pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Fine-tuning the model
optimizer = Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_dataset.shuffle(100).batch(8), epochs=10)

# New input for prediction
new_input = "I just watched a excellent and lovely movie!"

# Tokenize and encode the new input for the model
new_input_encodings = tokenizer.encode_plus(new_input, truncation=True, padding='max_length', max_length=128,
                                            return_tensors='tf')

# Predict using the fine-tuned model
# Pass the relevant tensor components (input_ids and attention_mask) to the model
predictions = model.predict({'input_ids': new_input_encodings['input_ids'],
                             'attention_mask': new_input_encodings['attention_mask']})

# Since the model outputs logits, apply a softmax to get probabilities
probabilities = tf.nn.softmax(predictions.logits)
print(probabilities)
# Determine the predicted class
predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]

# Output the result
sentiment = 'positive' if predicted_class == 1 else 'negative'
print(f"The sentiment of the review is: {sentiment}")
