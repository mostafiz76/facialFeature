# Face and facial and facial key points(eye,nose) detection algorithm, date 23rd oct 2018
import cv2          # importing openCV
import numpy as np  # importing numpy

# Creating the haar cascades for face,eyes and nose
face_cascade = cv2.CascadeClassifier('face_new.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade=cv2.CascadeClassifier('nose1.xml')

# Initializing webcam
cap = cv2.VideoCapture(0)
ds_factor = 0.5

# While the webcam is ON, reading every single frame and detecting face, eyes, nose
while True:
    ret, frame = cap.read() # Read every frame
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA) # Resizing the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # Converting the frame to gray scale
    faces = face_cascade.detectMultiScale(gray, 1.5)        # Detecting face
    print('Found {} faces!'.format(len(faces)))

    # For every detected face look for eyes and nose
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)   # Detecting eyes
        noses = nose_cascade.detectMultiScale(roi_gray) # Detecting nose

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (200, 255, 55), 1)

    cv2.imshow('Detector', frame)                  # Showing detected face, eye, nose

    c = cv2.waitKey(1)
    if c == 27:
        break


cap.release()
cv2.destroyAllWindows()