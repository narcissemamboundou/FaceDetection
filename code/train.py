
from keras.models import load_model
model = load_model('facenet_keras.h5')
import PIL
from os.path import isdir
from numpy import savez_compressed
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import urllib.request
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN

#test
print(model.inputs)
print(model.outputs)


path_data="/content/drive/My Drive/AI_BOOT_CAMP/images/5-celebrity-faces-dataset/data/"
train=path_data+"train/"
val=path_data+"val/"

X_train,y_train=load_dataset(train)
X_test,y_test=load_dataset(val)

# Save all images in path_save_data.npz
# @
np.savez_compressed(path_save_data,X_train=X_train,y_train= y_train,X_test=X_test,y_test=y_test)

#Load Xtrain, ytrain, Xtest, and ytest
data=load(path_save_data+".npz")

#implement the model facenet_kers
#@
facenet_kers="/content/facenet_keras.h5"
model = load_model(facenet_kers)

########## Function #############################

'''
To temporarily store the images locally for our analysis, we’ll retrieve each from its URL and write it to a local file.
Let’s define a function store_image for this purpose
'''

def store_image_with_link(url, local_file_name):
    with urllib.request.urlopen(url) as resource:
        with open(local_file_name, 'wb') as f:
            f.write(resource.read())


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


'''
detector = MTCNN()

faces = detector.detect_faces(image)
for face in faces:
    print(face)
rectangle_faces_selection('images', faces)

'''

def extract_face(img_path, required_size=(160, 160)):
       
        _img_=Image.open(filename)
        img=_img_.convert('RGB')	
        img_array=np.asanyarray(img)

        detector = MTCNN()	
        face_index=detector.detect_faces(img_array)	

        faces_in_image=[]

        for face in faces:

            # extract the bounding box from the requested face	
            x1,y1,width,height=face_index[0]['box']
            x2, y2=x1+width, y1+height		
            
            #construction of face
            face=img_array[y1:y2,x1:x2]

            #reformat the face tothe model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = np.asarray(image)

            faces_in_image.append(face_array)
            #plot the face 
        
        return (faces_in_image)

''' Test the function  extract_face
extracted_face = extract_face('iacocca_1.jpg')

# Display the first face from the extracted faces
plt.imshow(extracted_face[0])
plt.show()
'''


#@ Return faces list in a directory
#@ 
def upload_faces(directory):
        list_faces=[]
        for path in listdir(directory):
            img=Image.open(directory+path)
            img=img.convert('RGB')
            img_array=np.asanyarray(img)
            list_faces.append(img_array)
        return list_faces

''' Test - Function
a=upload_faces(link_ben_afflek)
for i in range(5):
  print(a[i])

'''
# return X and y - faces and labels for each subfolder
def load_dataset(directory):
        y = []
        X=[]
        for path in listdir(directory):
            new_dir=directory+path+'/'
            for i in upload_faces(new_dir):
                X.append(i)
                y.append(path)
        return (X,y)


# 
#@
def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(160, 160, 3),
      pooling='avg')

    # perform prediction
    return model.predict(samples)


'''
faces = [extract_face(image_path)
         for image_path in ['img_1.jpg', 'img_2.jpg']]

model_scores = get_model_scores(faces)

'''

def compare_two_faces(model_scores):
    result=""
    if cosine(model_scores[0], model_scores[1])<=0.4:
        result="Faces matched"
    else:
        result="Faces don't matched"
    return result

#return 2 lists with images 
#list1 with images of the first image
#list2 with images of the first image
#@ (list1, list2)

def compar_multi_faces( img1, img2):
    face_img1=extract_face(img1)
    face_img2=extract_face(img2)

    model_scores_1=model_scores(face_img1)
    model_scores_2=model_scores(face_img2)

    list_1=[]
    list_2=[]

    for id1, face_score_1 in enumerate(model_scores_1):
        for id2, face_score_2 in enumerate(model_scores_2):
            score = cosine(face_score_1, face_score_2)
            if score <=0.4:
                list_1.append(face_img1[id1])
                list_2.append(face_img2[id2])
    return (list_1,list_2)

''' ---------Test---------

compare=compar_multi_faces('img1.jpg','img2.jpg')
for i in range(len(compare[0])):
    plt.imshow(compare[0][i])
    plt.show()
    plt.imshow(compare[1][i])
    plt.show()

'''


