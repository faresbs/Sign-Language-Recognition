import numpy as np
import cv2
import sys

import model

video_capture = cv2.VideoCapture(0)

#Dimension of the rectangle for the bounding box
x = 100
y = 150
h = 200
w = 200


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (180,130)
fontScale = 3
fontColor = (255,255,255)
lineType = 2

#Image size
image_size = 224

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # draw rectangle on frame captured
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,185,0),3)

    #Extract bounding box image
    img = gray[y:y+h,x:x+w]
    cv2.imshow('image', img)

    #Resize captured image to be identical with the image size of the training data
    resized_image = cv2.resize(img,(image_size, image_size)) 
    
    #Predict with the model
    prediction = model.predict(resized_image)

    #Put text of the prediction on the image
    cv2.putText(frame, prediction, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    # Display the resulting frame
    cv2.imshow('Video Stream', frame)


    #PRESS Q TO QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()