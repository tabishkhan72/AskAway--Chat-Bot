import nltk
import json
import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Constants
INTENTS_FILE = 'intents.json'
WORDS_FILE = 'words.pkl'
CLASSES_FILE = 'classes.pkl'
MODEL_FILE = 'chatbot_model.h5'
HISTORY_FILE = 'training_history.pkl'
IGNORE_WORDS = ['?', '!', '.', ',']

lemmatizer = WordNetLemmatizer()

def load_intents(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Intents file not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def preprocess_intents(intents_data):
    words, classes, documents = [], [], []

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern.lower())
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in IGNORE_WORDS]
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save words and classes
    pickle.dump(words, open(WORDS_FILE, 'wb'))
    pickle.dump(classes, open(CLASSES_FILE, 'wb'))

    return words, classes, documents

def create_training_data(words, classes, documents):
    training_data = []
    output_template = [0] * len(classes)

    for tokens, tag in documents:
        bag = [1 if word in tokens else 0 for word in words]
        output_row = list(output_template)
        output_row[classes.index(tag)] = 1
        training_data.append((bag, output_row))

    random.shuffle(training_data)
    training_data = np.array(training_data, dtype=object)

    train_x = np.array(list(training_data[:, 0]))
    train_y = np.array(list(training_data[:, 1]))

    return train_x, train_y

def build_model(input_size, output_size):
    model = Sequential([
        Dense(128, input_shape=(input_size,), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_size, activation='softmax')
    ])

    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_chatbot():
    print("[INFO] Loading intents...")
    intents_data = load_intents(INTENTS_FILE)

    print("[INFO] Preprocessing data...")
    words, classes, documents = preprocess_intents(intents_data)

    print(f"[INFO] {len(documents)} training documents")
    print(f"[INFO] {len(classes)} classes: {classes}")
    print(f"[INFO] {len(words)} unique words")

    print("[INFO] Creating training data...")
    train_x, train_y = create_training_data(words, classes, documents)

    print("[INFO] Building model...")
    model = build_model(len(train_x[0]), len(train_y[0]))

    print("[INFO] Training model...")
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    print("[INFO] Saving model and training history...")
    model.save(MODEL_FILE)
    with open(HISTORY_FILE, 'wb') as f:
        pickle.dump(history.history, f)

    print("[SUCCESS] Chatbot model trained and saved successfully.")

if __name__ == "__main__":
    train_chatbot()
