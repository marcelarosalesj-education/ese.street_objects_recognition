"""
This program captures from the camera and converts the frame
into many different color spaces.

Resources:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    luv = cv2.cvtColor(frame, cv2.COLOR_BGR2Luv)

    # Display the resulting frame
    cv2.imshow('gray',gray)
    cv2.imshow('frame',frame)
    cv2.imshow('hsv',hsv)
    cv2.imshow('hls',hls)
    cv2.imshow('luv',luv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
