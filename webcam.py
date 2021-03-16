import cv2
import cvlib as cv
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

cap = cv2.VideoCapture(0) #this does not work on roboDK

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

while(True):
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    faces, confidences = cv.detect_face(img)

    # print(faces)

    for face in faces:
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        length = endX - startX
        height = endY - startY

        midX = int(startX + length/2)
        sqare_startX = int(midX - height/2)
        square_endX = int(midX + height/2)

        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(img, (sqare_startX,startY), (square_endX,endY), (0, 0, 0), 5) 

        face_crop = img[startY:endY, sqare_startX:square_endX, :]

        try:
            face = cv2.resize(face_crop, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
            PILImage = Image.fromarray(face)
            PILgrey  = PILImage.convert('L')
            face = np.array(PILgrey)

            face = face.reshape(-1, 100, 100, 1)
            face = face.astype("float32") / 255
            gender_prob = model_gender.predict(face)
            age_prob = model_age.predict(face)

            pred_age_index = np.argmax(age_prob)
            pred_age = pred_age_index + 1

            prob_m = gender_prob[0][0]
            prob_f = gender_prob[0][1]

            if prob_f > prob_m:
                label_string_gender = "GENDER: Female, " + str(round(prob_f * 100, 1)) + "%"
            else:  
                label_string_gender = "GENDER: Male, " + str(round(prob_m * 100, 1)) + "%"

            label_string_age = "AGE: " + str(pred_age) + ", " + str(round(age_prob[0][pred_age_index] * 100, 1)) + "%"

            image = cv2.putText(img, label_string_gender, (sqare_startX, startY - 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA) 
            image = cv2.putText(img, label_string_age, (sqare_startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA) 

            print("MALE: " + str(prob_m))
            print("FEMALE: " + str(prob_f))

            # img = face
        except cv2.error as e:
            print('Invalid frame!')



    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == 27:#ord('q'):
        break

cap.release()
cv2.destroyAllWindows()