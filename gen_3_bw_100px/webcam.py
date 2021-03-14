import cv2
import cvlib as cv
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

cap = cv2.VideoCapture(0) #this does not work on roboDK

with open('model.json', 'r') as json_file:
    json_savedModel= json_file.read()

model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.summary()
model_j.load_weights('model.h5')

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
        cv2.rectangle(img, (sqare_startX,startY), (square_endX,endY), (30,0,180), 5) 

        face_crop = img[startY:endY, sqare_startX:square_endX, :]


        try:
            face_cropped = cv2.resize(face_crop, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)

            PILImage = Image.fromarray(face_cropped)
            PILgrey  = PILImage.convert('L')
            face_bw= np.array(PILgrey)

            face_reshaped = face_bw.reshape(-1, 200, 200, 1)
            face_normalized_res = face_reshaped.astype("float32") / 255
            labels_pred = model_j.predict(face_normalized_res)

            prob_m = labels_pred[0][0]
            prob_f = labels_pred[0][1]

            if prob_f > prob_m:
                label_string = "Female, " + str(round(prob_f * 100, 2)) + "%"
            else:  
                label_string = "Male, " + str(round(prob_m * 100, 2)) + "%"

            image = cv2.putText(img, label_string, (sqare_startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 

            print("MALE: " + str(prob_m))
            print("FEMALE: " + str(prob_f))

            # img = face_bw
        except cv2.error as e:
            print('Invalid frame!')



    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == 27:#ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Hello World!")