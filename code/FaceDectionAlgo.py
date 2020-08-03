# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:31:27 2020

@author: n.mamboundou
"""
#importations

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as plot_img


famille="images/famille_1.jpg"

################ Réconnaissance Faciale avec la méthode Haar ############################

def HAAR_faces_detect(faces, image):
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class class_face_cascade:

    def __init__(self):
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    def detect_n_plot(self, im_path, scaleFactor=1.1, minNeighbors=3):
      image=cv2.imread(im_path,cv2.IMREAD_UNCHANGED)
      image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      faces = face_cascade.detectMultiScale(image_gray,scaleFactor, minNeighbors ) 
      return  HAAR_faces_detect(faces,image)
    
clf=class_face_cascade()
plt.imshow(clf.detect_n_plot(famille))

################ Réconnaissance Faciale avec la méthode LBP ############################

def HAAR_faces_detect(faces, image):
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class class_face_cascade:

    def __init__(self):
      face_cascade = cv2.CascadeClassifier('lbp_frontalface.xml')
    
    def detect_n_plot(self, im_path, scaleFactor=1.1, minNeighbors=3):
      image=cv2.imread(im_path,cv2.IMREAD_UNCHANGED)
      image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      faces = face_cascade.detectMultiScale(image_gray,scaleFactor, minNeighbors ) 
      return  HAAR_faces_detect(faces,image)

clf=class_face_cascade()
plt.imshow(clf.detect_n_plot(famille))

################ Comparaison entre La méthode Lbp et la méthode Haar ############################

class class_face_cascade:
    def __init__(self, lbp=False):
        if lbp:
            self.method = 'Lbp'
            face_cascade = cv2.CascadeClassifier("lbp_frontalface.xml")
        else:
            self.method = 'Haar'
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect_n_faces(self, im_path, scaleFactor=1.1, minNeighbors=3):
        im = cv2.imread(im_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        t1 = time.time()
        faces = face_cascade.detectMultiScale(im_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        t2 = time.time()
        fig, ax = plt.subplots()
        for coors in faces:
            ax.add_patch(Rectangle(coors[:2], width=coors[2], height=coors[3], color='red', fill=False))
        ax.imshow(im[:,:,[2,1,0]])
        plt.title(self.method + "'s detection time: {:.3f}".format(t2-t1))

object_face_cascade = class_face_cascade(lbp=True)
object_face_cascade.detect_n_faces(famille)

################ Réconnaissance Faciale avec la méthode Haar ############################




