import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random as rd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# import cv2
# import cvlib as cv

def draw_testdata(predicted_labels):
    random_array = rd.sample(range(1, X_test.shape[0]), 20)
    print(random_array)

    fig = plt.figure()
    # fig.suptitle("Test Accuracy: " + str(test_score) + ", Train Accuracy: " + str(train_score))

    i = 1
    for random in random_array:
        ax = fig.add_subplot(4,5,i)

        image = X_test[random]
        ax.imshow(image)
        ax.set_title("Label:" + str(y_test[random]))
        ax.axis('off')

        i = i+1
    plt.show()


rootdir = '../utkface'
max_iteration = 100000

path, dirs, files = next(os.walk(rootdir))
image_count = len(files)

if max_iteration < image_count:
    image_count = max_iteration

height = 200
length = 200
depth = 1

X = np.zeros((image_count+1, height, length))
y = np.zeros((image_count+1))

i = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # print(file)

        # Load Image to NP array
        if not file.endswith('.jpg'):
            continue
        im_iteration = Image.open(subdir + "/" + file).convert('L') #convert to grayscale
        im_iteration_array = tf.keras.preprocessing.image.img_to_array(im_iteration)

        # Normalize Image
        im_iteration_array = im_iteration_array.astype("float32") / 255
        im_iteration_array = im_iteration_array.reshape(200, 200)

        print(i)

        # Label 
        filename_array = file.split("_")
        age = filename_array[0]
        gender = filename_array[1]
        race = filename_array[2]
        # print("age: " + str(age) + ", gender: " + str(gender) + ", race: " + str(race))
        # print("")

        # Append to Dataset Matrix
        X[i] = im_iteration_array
        y[i] = gender

        i+=1
        if i > max_iteration:
            break

X = X.reshape(image_count+1, 200, 200, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#TRAINING
print("TRAINING")

num_classes = 2
input_shape = (200, 200, 1)

# draw_testdata(y_test)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

model.summary()

batch_size = 64
epochs = 50

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#Save Plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy: ' + str(score[1]))
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('performance_plot.png')