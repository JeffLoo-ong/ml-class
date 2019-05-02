from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

is_five_train = y_train == 5 # Changes the array of 0-9 to now 0-1
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height))) # Smush the 28x28 2d array into a single array
model.add(Dense(1, activation="sigmoid")) # Every one of the inputs to the dense layer will have a learned weight from the previous layer. A single perceptron is added in. Adding 2 would mean you're adding 2 layers
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, is_five_train, epochs=6, validation_data=(X_test, is_five_test),
          callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])
model.save('perceptron.h5')
