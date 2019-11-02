# 6.16, 6.17

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Flatten
# "2D Convolutional" layer and "Flatten" inputs to vectors

np.random.seed(123)
X = np.load('../generated_games/features-40k.npy')
Y = np.load('../generated_games/labels-40k.npy')

# input data shape is 3D, convert to 2D by doing 9x9x1
samples = X.shape[0]
size = 9
input_shape = (size, size, 1)

# Reshape input data
X = np.reshape(X, (samples, size, size, 1))

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

# Build a simple CNN for GO data with Keras
model = Sequential()
model.add(Conv2D(filters=48,                # First layer with 48 output filters
                 kernel_size=(3, 3),        # 3x3 convolutional kernel
                 activation='sigmoid',
                 padding='same',            # output is usually smaller than input,
                 input_shape=input_shape))  # pad matrix with 0's so input matches output

# Second layer, don't need filters or kernel_size
model.add(Conv2D(48, (3, 3), padding='same', activation='sigmoid'))
# Flatten the 3D output of previous layer
model.add(Flatten())
# Add two more dense layers
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(size * size, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=64,
          epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=64, epochs=15, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
