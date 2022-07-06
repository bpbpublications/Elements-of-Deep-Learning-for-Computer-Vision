# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:43:54 2020

@author: bhara
"""

import cv2
capture = cv2.VideoCapture(0)
while(True):
    result, frame = capture.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
capture.release()
cv2.destroyAllWindows()