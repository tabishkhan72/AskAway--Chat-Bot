import nltk
import json
import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download NLTK data if not already available
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_intents(file_path):
    lemmatizer = WordNetLemmatizer()
    words, classes, documents = [], [], []
    ignore_words = ['?', '!', '.', ',']

    # Load intents JSON file
    with open(file_path, 'r') as f:
        intents = json.load(f)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokenized_words = nltk.word_tokenize(pattern)
            words.extend(tokenized_words)
            documents.append((tokenized_words, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and sort
    words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
    classes = sorted(set(classes))

    # Save processed words and classes
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    return words, classes, documents

def create_training_data(words, classes, documents):
    lemmatizer = WordNetLemmatizer()
    training_data = []
    output_empty = [0] * len(classes)

    for doc_words, tag in documents:
        lemmatized_words = [lemmatizer.lemmatize(w.lower()) for w in doc_words]
        bag = [1 if w in lemmatized_words else 0 for w in words]

        output_row = list(output_empty)
        output_row[classes.index(tag)] = 1

        training_data.append((bag, output_row))

    # Shuffle and convert to NumPy arrays
    random.shuffle(training_data)
    training_data = np.array(training_data, dtype=object)

    train_x = np.array(list(training_data[:, 0]))
    train_y = np.array(list(training_data[:, 1]))
    return train_x, train_y

def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])

    # Use SGD optimizer
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def main():
    print("Preprocessing intents...")
    words, classes, documents = preprocess_intents('intents.json')

    print(f"{len(documents)} documents")
    print(f"{len(classes)} classes: {classes}")
    print(f"{len(words)} unique lemmatized words")

    print("Creating training data...")
    train_x, train_y = create_training_data(words, classes, documents)

    print("Building and training model...")
    model = build_model(len(train_x[0]), len(train_y[0]))
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Save model and training history
    model.save('chatbot_model.h5')
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("Model training complete and saved.")

if __name__ == "__main__":
    main()
