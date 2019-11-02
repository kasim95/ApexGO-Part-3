# CNN with Adagrad optimizer
import sys, os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

np.random.seed(123)
X = np.load('../generated_games/features-1000.npy')
Y = np.load('../generated_games/labels-1000.npy')

samples = X.shape[0]
size = 19
input_shape = (size, size, 1)

X = np.reshape(X, (samples, size, size, 1))

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

# Building a convolutional network for GO data with dropout and ReLUs
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Dropout(rate=0.5))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(size * size, activation='softmax'))
agrad = optimizers.Adagrad(0.01)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=agrad, metrics=['accuracy'])

# Evaluating your enhanced convolutional network
model.fit(X_train, Y_train, batch_size=64, epochs=200, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)

print('File name: ', os.path.basename(sys.argv[0]))
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
