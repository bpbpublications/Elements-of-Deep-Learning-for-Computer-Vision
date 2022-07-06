# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:38:46 2020

@author: bhara
"""
import cv2
capture = cv2.VideoCapture(r'C:\Users\bhara\OneDrive\Desktop\Book\images\sample-mp4-file.mp4')
while(capture.isOpened()):
    result, frame = capture.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
capture.release()
cv2.destroyAllWindows()