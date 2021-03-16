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

def draw_testdata():
    random_array = rd.sample(range(1, X_test.shape[0]), 20)
    print(random_array)

    fig = plt.figure()
    # fig.suptitle("Test Accuracy: " + str(test_score) + ", Train Accuracy: " + str(train_score))

    i = 1
    for random in random_array:
        ax = fig.add_subplot(4,5,i)

        age_prob = age_prob_matrix[random]
        pred_age_index = np.argmax(age_prob)
        pred_age = pred_age_index + 1
        label_age = "PRED. AGE: " + str(pred_age) + ", Real Age: " + str(y_test[random])

        gender_prob = gender_prob_matrix[random]
        prob_m = gender_prob[0]
        prob_f = gender_prob[1]
        if prob_f > prob_m:
            label_gender = "PRED. GENDER: Female"
        else:  
            label_gender = "PRED. GENDER: Male"

        image = X_test[random]

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ax.imshow(image)
        ax.set_title(label_age + "\n" + label_gender)
        ax.axis('off')

        i = i+1
    plt.show()

rootdir = 'utkface'
max_iteration = 1500

path, dirs, files = next(os.walk(rootdir))
image_count = len(files)

if max_iteration < image_count:
    image_count = max_iteration

height = 100
length = 100
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
        im_iteration_array = tf.keras.preprocessing.image.smart_resize(im_iteration_array, (100, 100), interpolation='bilinear' ) #Resize Image

        # Normalize Image
        im_iteration_array = im_iteration_array.astype("float32") / 255
        im_iteration_array = im_iteration_array.reshape(100, 100)

        # Label 
        filename_array = file.split("_")
        age = filename_array[0]
        gender = filename_array[1]
        race = filename_array[2]
        # print("age: " + str(age) + ", gender: " + str(gender) + ", race: " + str(race))
        # print("")

        # Append to Dataset Matrix
        X[i] = im_iteration_array
        y[i] = age

        i+=1
        if i > max_iteration:
            break

X = X.reshape(image_count+1, 100, 100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

with open('model_age.json', 'r') as json_file:
    json_savedModel= json_file.read()

#OPEN GENDER MODEL
with open('model_gender.json', 'r') as json_file:
    model_gender_json = json_file.read()

model_gender = tf.keras.models.model_from_json(model_gender_json)
model_gender.summary()
model_gender.load_weights('model_gender.h5')

#OPEN AGE MODEL
with open('model_age.json', 'r') as json_file:
    model_age_json = json_file.read()

model_age = tf.keras.models.model_from_json(model_age_json)
model_age.summary()
model_age.load_weights('model_age.h5')

gender_prob_matrix = model_gender.predict(X_test)
age_prob_matrix = model_age.predict(X_test)

draw_testdata()


