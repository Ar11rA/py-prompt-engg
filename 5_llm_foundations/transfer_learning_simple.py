import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

# Step 1: Create small datasets for movie and course reviews
# Creating dummy data for demonstration purposes only
# Movie reviews dataset (1: positive, 0: negative)
movie_reviews_data = np.random.randint(0, 2, (100, 10))  # 100 samples, 10 features
movie_reviews_labels = np.random.randint(0, 2, 100)      # 100 samples, binary labels

# Course reviews dataset (1: positive, 0: negative)
course_reviews_data = np.random.randint(0, 2, (100, 10))  # 100 samples, 10 features
course_reviews_labels = np.random.randint(0, 2, 100)      # 100 samples, binary labels

# Step 2: Train the model and fit it for course reviews
# Define a simple Sequential model
model = Sequential()
model.add(Embedding(50, 8, input_length=10))  # Assuming a vocabulary size of 50
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the course reviews dataset
model.fit(course_reviews_data, course_reviews_labels, epochs=10, verbose=1)

# Step 3: Write code to predict reviews for a new course
new_course_review = np.random.randint(0, 2, (1, 10))  # A new course review
new_course_prediction = model.predict(new_course_review)
print(f"Predicted sentiment for the new course review: {'Positive' if new_course_prediction[0][0] > 0.5 else 'Negative'}")

# Step 4: Use transfer learning and fine-tune this model to fit movie reviews
# Freeze the layers except the last one
for layer in model.layers[:-1]:
    layer.trainable = False

# Add a new dense layer for movie reviews
new_output_layer = Dense(1, activation='sigmoid')(model.layers[-2].output)
fine_tuned_model = Model(inputs=model.input, outputs=new_output_layer)

# Compile the fine-tuned model
fine_tuned_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model on the movie reviews dataset
fine_tuned_model.fit(movie_reviews_data, movie_reviews_labels, epochs=10, verbose=1)

# Step 5: Write code to predict reviews for a new movie
new_movie_review = np.random.randint(0, 2, (1, 10))  # A new movie review
new_movie_prediction = fine_tuned_model.predict(new_movie_review)
print(f"Predicted sentiment for the new movie review: {'Positive' if new_movie_prediction[0][0] > 0.5 else 'Negative'}")

