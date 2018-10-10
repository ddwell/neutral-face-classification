
# coding: utf-8

# # Adorable: Test task
# ### Vladimir Andryushechkin

# ## Tasks
# 
# - Программа, которая по набору путей до фотографий выводит 2 списка (по одному для каждого классификатора) фотографий, проходящих соответствующий фильтр. 
# 
# - Также требуется предоставить программный код (лучше в виде приватного репозитория на github),поясняющую документацию для запуска и выборки, использованные для обучения/валидации моделей.
# 
# - Описание того, какие метрики использовали при обучении/валидации классификаторов и почему, как бы вы подбирали параметры классификаторов в зависимости от возможных требований (например, при известной цене ошибки первого или второго рода).

# ## Extracting provided data

# In[262]:


import os

def loadData(cwd):
    Dataset, Data = {}, {}

    for root, dirs, files in os.walk(cwd):
        folders = dirs
        break
    #print(folders)

    for directory in folders:
        for root, dirs, files in os.walk( os.path.join( cwd, directory ) ):
            tmp_files = [os.path.splitext(file)[0] for file in files if '.jpg' in file or '.txt' in file]
            Dataset[directory] = tmp_files
            print('%s: %d files' % (directory,len(tmp_files)) )
            #print(tmp_files)

            if directory == 'landmarks':  
                for file in files:    
                    Data[os.path.splitext(file)[0]] = {}
                    if '.jpg' in file or '.txt' in file:
                        with open( os.path.join( cwd, directory, file), 'r') as f:
                            line = [x for i, x in enumerate(f) if i == 1][0].replace('\n','').split(' ')
                            line = [[float(x) , float(y)] for x, y in zip(line[::2], line[1::2]) if x != '' and y != '']

                            Data[os.path.splitext(file)[0]]['landmarks'] = line

    for file in Data:
        for directory in ['open_mouth','smile']:
            Data[file][directory] = 0
            if file in Dataset[directory]:
                Data[file][directory] = 1       
                
    return Data, Dataset

cwd = os.path.abspath( os.path.join( os.getcwd(),  'example_data' ) )
Data, Dataset = loadData(cwd)


# ## Extracting features from images
# Landmarks are uniformly scaled respectively to the face's rectangle

# In[268]:


# this code is partially from DLIB examples
#
# http://dlib.net/face_landmark_detection.py.html
#
#

import sys
import os
import dlib
import glob

def convertScale(coord, source=(0,100), target=(0,1000)):
    z = (coord - source[0]) / (source[1]-source[0])
    return z*target[1]

Data2 = {}
"""
if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()
"""
predictor_path = os.path.abspath( os.path.join( os.getcwd(),  'files', 'shape_predictor_68_face_landmarks.dat' ) )#sys.argv[1]
faces_folder_path = os.path.abspath( os.path.join( os.getcwd(),  'example_data', 'images' ) )#sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    filename = os.path.splitext(os.path.basename(f))[0]
    img = dlib.load_rgb_image(f)

#     win.clear_overlay()
#     win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    #print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
#         print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
#                                                   shape.part(1)))
        # Draw the face landmarks on the screen.
#         win.add_overlay(shape)

#     win.add_overlay(dets)
    dlib.hit_enter_to_continue()
    
    # feature normalization
    
    cx = (shape.rect.tl_corner().x, shape.rect.br_corner().x)
    cy = (shape.rect.tl_corner().y, shape.rect.br_corner().y)

    vec = np.empty([68, 2], dtype = float)
    for b in range(shape.num_parts):
        vec[b] = [convertScale(shape.part(b).x, source = cx), convertScale(shape.part(b).y, source = cy)]
    
    Data2[filename] = {'landmarks':vec, 'open_mouth':Data[filename]['open_mouth'], 'smile':Data[filename]['smile']}


# ### Feature normalization
# is needed to uniform feature vectors 
# Pictures with different sizes and proportions are not uniform
# 
# Eg:
# - big image and small face on it
# - small image and full-screened face on it
# 
# From the entire image only the face rectangle is required

# In[281]:


get_ipython().run_line_magic('pylab', 'inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime

from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
num='00027'

x = [d[0] for d in Data2[num]['landmarks']]
y = [d[1] for d in Data2[num]['landmarks']]
 
# plt.grid(True)
line_0, = plt.plot(x,y, 'b.',  label='landmarks')
plt.legend(handles=[line_0])
plt.gcf().autofmt_xdate()
plt.ylabel('Y')
plt.xlabel('X')
plt.title("face lanmarks on "+num)

plt.grid(linestyle='-', linewidth='0.0')
plt.minorticks_on()

plt.show();


# ## Feature and label engineering

# In[299]:


# feature vectors
X = [list(chain(*Data2[x]['landmarks'])) for x in Data2]

# label vectors
y_open_mouth = [Data2[y]['open_mouth'] for y in Data2]
y_smile = [Data2[y]['smile'] for y in Data2]


# #### Distribution of classes / labels

# In[300]:


def printLabelDistribution(y, name=''):
    counter = Counter(y)
    print("'%s' label distribution:\n\tnegative: %d\n\tpositive: %d\n" % (name, counter[0], counter[1]))

printLabelDistribution(y_open_mouth, 'open_mouth')
printLabelDistribution(y_smile, 'smile')


# In[624]:


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import natsort

# dimension = EMOTION

s = y_smile#[:, dimension] #y_annotated[dimension]
order = sorted(range(len(s)), key=lambda k: s[k])

g1 = y_open_mouth
g2 = y_smile
g3 = [s+om for s,om in zip(y_smile, y_open_mouth)]

line_0, = plt.plot(np.array(g1)+0.01, '.',  label='open_mouth')
line_1, = plt.plot(np.array(g2)-0.01, '.', label='smile')
line_2, = plt.plot(np.array(g3), '.', label='both')
plt.grid(True)
plt.legend(handles=[line_0, line_1, line_2])
plt.legend(bbox_to_anchor=(1.02, .4, .65, .0), loc=3,ncol=1, mode="expand", borderaxespad=1.0)

plt.xlabel('images')
plt.title("Classes distribution")
plt.show()


# ### Oversampling
# is advised due to uneven distribution of classes in order to have better accuracy

# In[63]:


from imblearn.over_sampling import RandomOverSampler

def oversampling(X, y):
    ros = RandomOverSampler(ratio='auto')
    X_resampled, y_resampled = ros.fit_sample(X,y)
    
    print("dataset oversampled:\t %d -> %d" % ( len(y), len(y_resampled)))    
    return(X_resampled, y_resampled)

X_smile_resampled, y_smile_resampled = oversampling(X, y_smile)
X_open_mouth_resampled, y_open_mouth_resampled = oversampling(X, y_open_mouth)
# X_som_resampled, y_som_resampled = oversampling(X, [(i,j) for i,j in zip(y_open_mouth,y_smile)] )


# ### Train / Test split

# In[509]:


X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X, y_smile, test_size=0.25, random_state=42)
X_om_train, X_om_test, y_om_train, y_om_test = train_test_split(X, y_open_mouth, test_size=0.25, random_state=42)


# In[510]:


X_s_rsmp, y_s_rsmp = oversampling(X_s_train, y_s_train)
X_s_test_rsmp, y_s_test_rsmp = oversampling(X_s_test, y_s_test)

X_om_rsmp, y_om_rsmp = oversampling(X_om_train, y_om_train)
X_om_test_rsmp, y_om_test_rsmp = oversampling(X_om_test, y_om_test)


# ## SVM Training (baseline model)
# ### Parameter tuning
# 

# In[283]:


from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer


# In[541]:


c_array = [0.00001,0.0001,0.001,0.01,0.1,1.0]

print('smile')
for c_val in c_array:
    cvs_s = cross_val_score(estimator = LinearSVC(), X=X, y=y_smile, cv=5, n_jobs=5, scoring='f1')
    print(c_val, np.mean(cvs_s), cvs_s)

print('open_mouth')
for c_val in c_array:
    cvs_s = cross_val_score(estimator = LinearSVC(), X=X, y=y_open_mouth, cv=5, n_jobs=5, scoring='f1')
    print(c_val, np.mean(cvs_s), cvs_s)


# In[542]:


svc_s_trained = LinearSVC(C=0.1, random_state=0)
svc_s_trained.fit(X_s_rsmp, y_s_rsmp)

f1_score(y_s_test_rsmp , svc_s_trained.predict(X_s_test_rsmp)), f1_score(y_s_test , svc_s_trained.predict(X_s_test))


# In[543]:


svc_om_trained = LinearSVC(C=0.0001, random_state=0)
svc_om_trained.fit(X_om_rsmp, y_om_rsmp)

f1_score(y_om_test_rsmp , svc_om_trained.predict(X_om_test_rsmp)), f1_score(y_om_test , svc_om_trained.predict(X_om_test))


# ## Used metrics:
# - Dataset is split on TRAIN(75%) and TEST(25%) sets
# 
# - OVERSAMPLING is used due to the classes imbalance
# 
# - 5-fold CROSS-VALIDATION is performed on the train set in order to tune SVM's parameters
# 
# - F1 SCORING is used to test model's performane on the TEST set

# ## Final training and Model saving

# In[561]:


from sklearn.externals import joblib

def checkFolder(filename):
    dir = os.path.dirname(filename)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir) 

def saveModelFor(model, name, path):
    checkFolder(path)
    filename = os.path.join(path, 'svc_'+name+'.dump')
    _ = joblib.dump(model, filename, compress=9)
    print("'%s' model saved to <%s>" % (name,filename))
    
    
# svcTrained.fit(svc_X_train_oversampled, svc_y_train_oversampled)
# print(svcTrained)
save_path = os.path.abspath( os.path.join( os.getcwd(),  'files' ))


saveModelFor(model=svc_s_trained, name='smile', path = save_path)
saveModelFor(model=svc_om_trained, name='open_mouth', path = save_path)


# ## adorable-neutral-face-classification.py

# In[626]:


import sys
import os
import dlib
import glob

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
faces_folder_path = os.path.abspath( os.path.join( os.getcwd(),  'test_data' ) )#sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
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


# In[629]:


# result printing
print("%s\t%s\t%s" % ("open_mouth","smile","image"))

for v0,v1,v2 in zip(Data3,_y_open_mouth,_y_smile):
    print(" %d\t\t %d\t%s.jpg" % (v1,v2,v0))


# ### Investigation

# In[612]:


import cv2
import numpy as np

img = cv2.imread(os.path.abspath( os.path.join( os.getcwd(),  'test_data','00800.jpg' )))
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,img)

cv2.imwrite(os.path.abspath( os.path.join( os.getcwd(),  'sift_keypoints.jpg' )),img)

