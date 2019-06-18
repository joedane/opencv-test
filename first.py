#! /usr/bin/python3

import numpy as np
import cv2 as cv


def detectAndDisplay(frame, cascade, save):
    frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_grey = cv.equalizeHist(frame_grey)
    
    faces = cascade.detectMultiScale(frame_grey)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 3)
    
    cv.imshow('frame', frame)
    
    if save:
        cv.imwrite('save.jpg', frame)



cap = cv.VideoCapture('mcem0_head.mpg')
cascade = cv.CascadeClassifier()
if not cascade.load(cv.samples.findFile('../opencv/data/haarcascades/haarcascade_frontalface_alt.xml')):
    print('error loading classifier')
    exit(-1)

while cap.isOpened():
    ret, frame = cap.read()
    save = False

    if not ret:
        print("XXXX")
        break

    key = cv.waitKey(10)
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        save = True

    detectAndDisplay(frame, cascade, save) 
   
cap.release()
cv.destroyAllWindows()

