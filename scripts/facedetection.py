# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 02:51:29 2020

@author: bhara
"""

import cv2
detection_xml = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
image = cv2.imread(r'C:\Users\bhara\OneDrive\Desktop\Book\images\geek.jpg')
grey = cv2.cvtColor(image, 0)
facedetection = detection_xml.detectMultiScale(grey, 1.5, 4)
for (x,y,w,h) in facedetection:
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\bhara\OneDrive\Desktop\Book\images\facedetected.jpg',image)