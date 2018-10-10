import sys
import os
import dlib
import glob

from sklearn.externals import joblib
from sklearn.svm import LinearSVC

import numpy as np
from itertools import chain

def convertScale(coord, source=(0,100), target=(0,1000)):
    z = (coord - source[0]) / (source[1]-source[0])
    return z*target[1]

def _load_classifier(PATH, NAME):
    filename = os.path.join(PATH, 'svc_'+NAME+'.dump')
    model = joblib.load(filename)
    return model

model_path = os.path.abspath( os.path.join( os.getcwd(),  'files' ))
models = {}

for est in ['open_mouth','smile']:
    models[est] = _load_classifier(model_path, est)

Data3 = {}

predictor_path = os.path.abspath( os.path.join( os.getcwd(),  'files', 'shape_predictor_68_face_landmarks.dat' ) )#sys.argv[1]
faces_folder_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

    filename = os.path.splitext(os.path.basename(f))[0]
    img = dlib.load_rgb_image(f)
    
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        
    #dlib.hit_enter_to_continue()
    
    # feature normalization
    
    cx = (shape.rect.tl_corner().x, shape.rect.br_corner().x)
    cy = (shape.rect.tl_corner().y, shape.rect.br_corner().y)

    vec = np.empty([68, 2], dtype = float)
    for b in range(shape.num_parts):
        vec[b] = [convertScale(shape.part(b).x, source = cx), convertScale(shape.part(b).y, source = cy)]
    
    Data3[filename] = {'landmarks':vec}  
    
    
# feature vectors
_X = [list(chain(*Data3[x]['landmarks'])) for x in Data3]

# classification
_y_open_mouth = models['open_mouth'].predict(_X)
_y_smile = models['smile'].predict(_X)

#result printing
sys.stdout.write("%s\t%s\t%s\n" % ("open_mouth","smile","image"))

for v0,v1,v2 in zip(Data3,_y_open_mouth,_y_smile):
    sys.stdout.write(" %d\t\t %d\t%s.jpg\n" % (v1,v2,v0))