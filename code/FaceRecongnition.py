
import PIL
import urllib.request
from os.path import isdir
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
from numpy import savez_compressed
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder



'''

L'objectif de cette partie est d'arriver a comparer des visages 
pour déterminer s'il sagie d'une meme personne ou non!
(1) detection du visage avec MTCNN
(2) Identification faciale avec VGGFace et resnet50
(3) Verification d'identité avec VGGFace (cosine)

'''


########################## Fonctions #########################

# Fonction qui permet de tellecharger temporairement une image via son url
# @  
def tellecharger_image(url, nom_fichier):
    with urllib.request.urlopen(url) as resource:
        with open(nom_fichier, 'wb') as f:
            f.write(resource.read())

# Fonction qui permet de desiner un carré rouge autour d'un ou plusieurs visages
#@ 

def detecter_visage_rectangle(lien_image,visage_list):
    #lecture de l'image
    image=plt.imread(lien_image)
    #on l'affiche
    plt.imshow()
    
    ax=plt.gca()

    for visage in visage_list:
        x, y, largeur, hauteur = visage['box']
        rectangle = Rectangle((x, y), larg, hauteur,
                          fill=False, color='red')
        ax.add_patch(rectangle)
    plt.show()

# Fonction qui permet de desiner un carré rouge autour d'un ou plusieurs visages en rajoutants 
#des points autour des yeux et la bouhe et sur le nez
#@ 

def detecter_visage_rectangle_points(lien_image,visage_list):
    #lecture de l'image
    image=plt.imread(lien_image)
    #on l'affiche
    plt.imshow()
    
    ax=plt.gca()

    for visage in visage_list:
        x, y, largeur, hauteur = visage['box']
        rectangle = Rectangle((x, y), larg, hauteur,
                          fill=False, color='red')
        ax.add_patch(rectangle)

        for _, details in visage['keypoints'].items():
            points=Circle(details, radius=2, color='red')
            ax.add_patch(points)

    plt.show()


#Fonction qui extrait un ou plusieur image d'une photo avec une taille définie
#@ liste contenant des visages sous forme de matrice

def extraire_visages(lien_visage, dimensions=(160,160) ):
        
        image=Image.open(lien_visage)
        image=image.convert('RGB')	
        pixels=np.asanyarray(image)

        detecteur = MTCNN()	
        visage_liste=detector.detect_visages(pixels)	

        liste=[]

        for visage in visage_liste:

            # extract the bounding box from the requested visage	
            x1,y1,largeur,hauteur=visage[0]['box']
            x2, y2=x1+largeur, y1+hauteur		
            
            #construction du visage
            visage=pixel[y1:y2,x1:x2]

            #reformat the visage tothe model size
            image = Image.fromarray(visage)
            image = image.resize(dimensions)
            pixels = np.asarray(image)

            liste.append(pixels)
          
        return liste


# Fonction qui extrait un ou plusieurs images et calcule leurs score
#@ prédiction
def model_scores(lien_images):

    visages=[extraire_visages(v) for f in lien_images]
    pixels = asarray(visages, 'float32')

    pixels = preprocess_input(pixels, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(160, 160, 3),
      pooling='avg')
    #  prediction
    return (model.predict(pixels))

# Détermine si un nouveau visage est identique à un visage connu
#@ YES or NO

def verification(visage_connu, visage_inconnu):
    result=""
    if cosine(visage_connu, visage_inconnu)<=0.4:
        result="Yes" 
    else:
        result="NO"
    return result

