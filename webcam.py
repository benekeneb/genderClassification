import cv2

cap = cv2.VideoCapture(0) #this does not work on roboDK
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#cap = cv2.VideoCapture('/Users/riddhamanna/Documents/RoboDK/ymaze.mp4') #This works on roboDK

while(True):
    ret, img = cap.read()
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == 27:#ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Hello World!")