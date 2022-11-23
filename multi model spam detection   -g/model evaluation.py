import tensorflow as tf
import io
import requests
from PIL import Image
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from tensorflow.keras.models import load_model
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import collections
import contractions
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import random
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub



total_images = []



path = r"D:\SPAM DATASET\new_dataset\train\train_ham"
folder_list = os.listdir(path)

for item in folder_list:
    filename = os.path.join(path,item)
    k = [filename,0]
    total_images.append(k)
    
    
    
    
total_images = pd.DataFrame(total_images, columns = ['Name', 'class'])

    
    
    # shuffle the DataFrame rows
total_images = total_images.sample(frac = 1)


  
        


def eval_list(X_train,X_test,y_train,y_test):
            

    return my_reloaded_model.score(X_test,y_test)
    
    
    






    
    
import numpy as np

X = total_images.iloc[:,0]
y = total_images.iloc[:,1]
X = X.to_numpy()
y = y.to_numpy()


for i in range(len(X)):
    try:
        test_img = image.load_img(X[i],target_size=(224,224))
        test_img = image.img_to_array(test_img)
        test_img = np.array([test_img])
        test_img = test_img/255
        X[i] = test_img
    except:
        print("found broken data stream")
    
    



from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5)



scores = []

i = 1
for train_index, test_index in kfold.split(X,y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], \
                                       y[train_index], y[test_index]
    #scores.append(get_score(X_train, X_test, y_train, y_test))
    # eval_list = get_score(X_train,X_test,y_train,y_test)
    # scores.append(eval_list)
    
    i+=1
    

pip install --upgrade tensorflow-estimator==2.3.0
import tensorflow_hub as hub


os.chdir(r"D:\SPAM DATASET\models\\")

my_reloaded_model = load_model('cnn_mobilenet_success_AUGDATASET1.h5',custom_objects={'KerasLayer':hub.KerasLayer})
my_reloaded_model.fit(X_train,y_train) 
    
    