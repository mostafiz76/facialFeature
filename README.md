# Facial key points detection

This project will take video input from webcam, convert every frame into gray scale image, search for any face/faces present in the frame. After one or more faces are detected the algorithm will search for eyes and nose in every detected face.

The same project can be use to read a single image and detect the facial keypoints.

### Tech

the project uses a number of technologies to work properly:

* openCV- version 3.4.3
* Python- version 3.7.0
* PyCharm- edition 2018.2.8
* Numpy- version 1.15.3
* Windows 10 
### Installation

We will start by installing the latest stable version of Python 3, I used python 3.7. Head over to https://www.python.org/downloads/ and download the installer. The default Python Windows installer is 32 bits and I used the same. If your machine is 64bit, you need to install accordingly.

Start the installer and select Customize installation. On the Advanced Options screen make sure to check Install for all users, Add Python to environment variables and Precompile standard library click the install button. It will take few minutes. To check if the installation worked correctly open windows command promt and type 'python' it should show the python version.

As OpenCV for Windows binary from http://opencv.org/ does not have the required pyd file for Python 3, we need to install a wheel package. Download the Numpy version corresponding to your Python installation from https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy. I installed numpy-1.15.3+mkl-cp37-cp37m-win32.whl .

Download the OpenCV version corresponding to your Python installation from https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv. I installed opencv_python-3.4.3+contrib-cp37-cp37m-win32.whl .

Now open cmd, goto downloads and give the following commands to install numpy and opencv
```sh
pip install numpy-1.15.3+mkl-cp37-cp37m-win32.whl
pip install opencv_python-3.4.3+contrib-cp37-cp37m-win32.whl
```
To check the installation open a python interpreter(I use pycharm) and check the openCv version with the bellow  commands-
```sh
import cv2
print(cv2.__version__)
```




### Code Details
I used PyCharm to write and debug the project code.

At first we need to initialize the Haar cascades to detect face,eyes and nose. We need pre-created xml files which have tranning datas for face, eyes and nose.

```sh
face_cascade = cv2.CascadeClassifier('face_new.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade=cv2.CascadeClassifier('nose1.xml')
```
 initialize the webcam
 ```sh
 cap = cv2.VideoCapture(0)
 ```
 
 While the webcam is on, check in every frame for faces. Once face is detected search for eyes and nose.
 
 ```sh
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
 ```
 
 Release the camera and destroy all windows
 ```sh
 cap.release()
cv2.destroyAllWindows()
 ```
 

#### Building for source
run the project from PyCharm itself or esle you can run the myCode.py from the cmd.
