# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:55:55 2020

@author: n.mamboundou
"""

import cv2


#video object 
video = cv2.VideoCapture(0)

a=0

while True:
    a=a+1
    check, frame=video.read()
    
    print(check)
    print(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #show the frame 
    cv2.imshow("Face detection", gray)
    
    key=cv2.waitKey(1)
    
    if key==ord('q'):
        break
    
print(a)
 #shutdown the camera
video.release()
cv2.destroyAllWindows
    