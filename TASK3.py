from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras import datasets
from keras.optimizers import Adam
# TASK 3

# Load data
# EMNIST Letters dataset, 27 classes [0-9A-Z] (class 0 = N/A)
(X_train, y_train), (X_test, y_test) = datasets.emnist.load_data(type='letters')

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 27)
y_test = to_categorical(y_test, 27)

# Build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(27, activation='softmax'))  # 27 classes

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Save model
model.save('emnist_letters_model.h5')

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make a prediction on the first 10 test images
predictions = model.predict(X_test[:10])
print('Predictions on the first 10 test images:', predictions.argmax(axis=1))