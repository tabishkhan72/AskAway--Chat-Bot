import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Initialize lemmatizer and required lists
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Load and parse intents file
with open('intents.json') as file:
    intents = json.load(file)

# Tokenize, lemmatize, and collect words, classes, and documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)
        documents.append((tokenized_words, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Clean up and sort words and classes
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words")

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data (Bag of Words + One-hot encoded labels)
training_data = []
output_empty = [0] * len(classes)

for doc_words, tag in documents:
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in doc_words]
    bag = [1 if word in lemmatized_words else 0 for word in words]
    
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    
    training_data.append((bag, output_row))

# Shuffle and convert to NumPy arrays
random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)

train_x = np.array(list(training_data[:, 0]))
train_y = np.array(list(training_data[:, 1]))

print("Training data created")

# Build the model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
optimizer = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model and training history
model.save('chatbot_model.h5')
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Model training complete and saved.")
