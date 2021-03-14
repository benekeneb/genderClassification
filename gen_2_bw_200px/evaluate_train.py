import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random as rd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import cvlib as cv

def draw_testdata(predicted_labels):
    random_array = rd.sample(range(1, X_test.shape[0]), 20)
    print(random_array)

    fig = plt.figure()
    # fig.suptitle("Test Accuracy: " + str(test_score) + ", Train Accuracy: " + str(train_score))

    i = 1
    for random in random_array:
        ax = fig.add_subplot(4,5,i)

        label = y_test[random]
        if label == 0.0:
            label_string = "Männlich"
        else:
            label_string = "Weiblich"

        image = X_test[random]

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ax.imshow(image)
        ax.set_title(label_string)
        ax.axis('off')

        i = i+1
    plt.show()

rootdir = '../utkface'
max_iteration = 1500

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
        print(file)

        # Load Image to NP array
        if not file.endswith('.jpg'):
            continue
        im_iteration = Image.open(subdir + "/" + file).convert('L') #convert to grayscale
        im_iteration_array = tf.keras.preprocessing.image.img_to_array(im_iteration)

        # Normalize Image
        im_iteration_array = im_iteration_array.astype("float32") / 255
        im_iteration_array = im_iteration_array.reshape(200, 200)

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

with open('model.json', 'r') as json_file:
    json_savedModel= json_file.read()

#load the model architecture 
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.summary()
model_j.load_weights('model.h5')

labels_pred = model_j.predict(X_test)

draw_testdata(labels_pred)


