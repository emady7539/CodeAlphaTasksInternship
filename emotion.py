import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

# TASK 2




# Load data
def load_data(file_paths, labels):
    data = []
    for file_path in file_paths:
        # Load audio file
        y, sr = librosa.load(file_path)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        data.append(mfcc)
    return np.array(data), np.array(labels)

file_paths = [...]  # list of file paths to your audio files REPLACE THIS WITH YOUR FILE PATHS
labels = [...]  # corresponding labels REPLACE THIS WITH YOUR LABELS 
data, labels = load_data(file_paths, labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Reshape data for CNN model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # number of unique emotions
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(np.unique(labels)))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(np.unique(labels)))

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_split=0.2)

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])