

import cv2


'''
img =plot_img.imread("test.jpg")
plt.imshow(img,cmap="gray")

img_grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_grey, cmap="gray")



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img_grey)

print("the number of faces :", len(faces))

for (x,y,w,h) in faces:
     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
     
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img = cv2.imread("test.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray_img,1,1,4)
print("the number of faces :", len(faces))
for (x,y,w,h) in faces:
     cv2.rectangle(gray_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
     


'''

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
cap.release()

'''
while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
'''        
# Release the VideoCapture object
def rectangle_faces_selection(image_path, faces):
      # display image
    image = plt.imread(image_path)
    plt.imshow(image)

    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                          fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()




