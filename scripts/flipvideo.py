# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:58:42 2020

@author: bhara
"""

import cv2
capture = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter('output.avi',codec, 60.0, (640,480),0)
while(capture.isOpened()):
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outvideo.write(gray)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

capture.release()
outvideo.release()
cv2.destroyAllWindows()