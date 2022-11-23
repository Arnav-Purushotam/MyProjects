#coding the fusion model
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
import nltk
nltk.download('averaged_perceptron_tagger')
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import string

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve




def create_fusion_model():
    model = Sequential()
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    return model





def activate_fusion_model(final_list_train,y_pred,final_list_test,y_test,fold):
    model = create_fusion_model()
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #total_y = np.concatenate((y_pred,y_test),axis=0)
    history = model.fit(x = final_list_train, validation_data=(final_list_test, y_test), y = y_pred, epochs = 32)
    link_to_save = r'D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\1st type of dataset - only images fusion models\fusion_model_fold' + str(fold) + '.h5'
    if os.path.isfile(link_to_save) is False:
        model.save(link_to_save)
    
    #saving history as a dict
    #history object holds different training metrics spanned accross every training epoch
    link_to_save_history = r'D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\1st type of dataset - only images fusion models\fusion_modelHistory_fold' + str(fold)
    with open(link_to_save_history,"wb") as file_pi:
        pickle.dump(history.history,file_pi)
        
    return link_to_save,link_to_save_history










def data_creator(model_path,data_path_train_spam,data_path_train_ham):
    
    fusion_list = []
    #new_cnn_model = load_model(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold1.h5")
    new_cnn_model = load_model(model_path)
    #image_path = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold1\test\combined_test_spam_ham"
    image_path_spam = data_path_train_spam
    image_list_spam = os.listdir(image_path_spam)
    i = 0
    y = []
    for test_img in image_list_spam:
        if i%10 != 0:
            test_img = image.load_img(os.path.join(image_path_spam,test_img),target_size=(224,224))
            test_img = image.img_to_array(test_img)
            test_img = np.array([test_img])
            test_img = test_img/255
            test_img = test_img.reshape(1,224,224,3)
            result = new_cnn_model.predict(test_img)
            #fusion_list contains the probability os spam
            fusion_list.append([result[0][1]])
            y.append(1)
            print(i)
        else:
            fusion_list.append([0.5])  
            y.append(0.5)
        i+= 1
       
        
       
    i=0
    image_path_ham = data_path_train_ham
    image_list_ham = os.listdir(image_path_ham)        
    for test_img in image_list_ham:
        if i%10 != 0:
            test_img = image.load_img(os.path.join(image_path_ham,test_img),target_size=(224,224))
            test_img = image.img_to_array(test_img)
            test_img = np.array([test_img])
            test_img = test_img/255
            test_img = test_img.reshape(1,224,224,3)
            result = new_cnn_model.predict(test_img)
            #fusion_list contains the probability os spam
            fusion_list.append([result[0][1]])
            y.append(0)
            print(i)
        else:
            fusion_list.append([0.5])  
            y.append(0.5)
        i+= 1

    

    
    return fusion_list,y
        

listf_train,y_train = data_creator(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold5.h5",r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold5\train\spam",r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold5\train\ham")
df1 = pd.DataFrame(data = listf_train,columns = ['probability-image'])
df2 = pd.DataFrame(data = y_train,columns = ['y-image'])
df_train = pd.concat([df1,df2],axis=1)
df_train = df_train.sample(frac=1)
images_df_train = df_train

listf_test,y_test = data_creator(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold1.h5",r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold1\test\spam",r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold1\test\ham")
df1 = pd.DataFrame(data = listf_test,columns = ['probability-image'])
df2 = pd.DataFrame(data = y_test,columns = ['y-image'])
df_test = pd.concat([df1,df2],axis=1)
df_test = df_test.sample(frac=1) 

# images_df_train = df_train
images_df_test = df_test





def remove_hyperlink(word):
    return  re.sub(r"http\S+", "", word)

def to_lower(word):
    result = word.lower()
    return result

def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result


def remove_punctuation(word):
    
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_whitespace(word):
    result = word.strip()
    return result

def replace_newline(word):
    return word.replace('\n','')



def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation,remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence














df_train = pd.read_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold1\fold1_train.csv",encoding = "'latin'")
df_test = pd.read_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold1\fold1_test.csv" ,encoding = "'latin'")


df_train = df_train.iloc[:,:2]
df_train.columns = ["spam","text"]

df_test = df_test.iloc[:,:2]
df_test.columns = ["spam","text"]




emails_train = df_train["text"]
emails_test = df_test["text"]
target_train = df_train["spam"]
target_test = df_test["spam"]


x_train = [clean_up_pipeline(o) for o in emails_train]
x_test = [clean_up_pipeline(o) for o in emails_test]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_y = le.fit_transform(target_train.values)
test_y = le.transform(target_test.values)




## some config values 
embed_size = 100 # how big is each word vector
max_feature = 7836 # how many unique words to use (i.e num rows in embedding vector)
max_len = 79 # max number of words in a question to use





from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

x_train_features = np.array(tokenizer.texts_to_sequences(x_train))
x_test_features = np.array(tokenizer.texts_to_sequences(x_test))


x_train_features = pad_sequences(x_train_features,maxlen=max_len)
x_test_features = pad_sequences(x_test_features,maxlen=max_len)




new_lstm_model = load_model(r'D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\models\fold1.h5')

listg_train = []
y_train = []
for j in range(len(x_train_features)):
    if j%10 != 0:
        my_input = x_train_features[j].reshape(1,79)
        kprobs = new_lstm_model.predict(my_input)
        # yhat_classes.append(np.argmax(kprobs[0]))
        listg_train.append(kprobs[0][0])
        y_train.append(train_y[j])
        j+=1
    else:
        listg_train.append(0.5)
        y_train.append(0.5)
        j+=1
        
    
df1 = pd.DataFrame(data = listg_train,columns = ['probability-text'])
df2 = pd.DataFrame(data = y_train,columns = ['y-text'])
df_train = pd.concat([df1,df2],axis=1)
df_train = df_train.sample(frac=1)
text_df_train = df_train


listg_test = []
y_test = []
for j in range(len(x_test_features)):
    if j%10 != 0:
        my_input = x_test_features[j].reshape(1,79)
        kprobs = new_lstm_model.predict(my_input)
        # yhat_classes.append(np.argmax(kprobs[0]))
        listg_test.append(kprobs[0][0])
        y_test.append(test_y[j])
        j+=1
    else:
        listg_test.append(0.5)
        y_test.append(0.5)
        j+=1
    
df1 = pd.DataFrame(data = listg_test,columns = ['probability-text'])
df2 = pd.DataFrame(data = y_test,columns = ['y-text'])
df_test = pd.concat([df1,df2],axis=1)
df_test = df_test.sample(frac=1)
text_df_test = df_test





images_df_test = images_df_test.values
images_df_train = images_df_train.values
text_df_test = text_df_test.values
text_df_train = text_df_train.values



train_list_x = []
train_list_y = []
for j in range(4456):
    
    if(images_df_train[j][1]==1 or text_df_train[j][1]==1):
        train_list_y.append(1)
    elif(images_df_train[j][1]==0.5 and text_df_train[j][1]==0.5):
        continue
    else:
        train_list_y.append(0)    

    train_list_x.append([images_df_train[j][0],text_df_train[j][0]])
    

 
for i in range(4010):


       


test_list_x = []
test_list_y = []
for j in range(1111):
    
    if(images_df_test[j][1]==1 or text_df_test[j][1]==1):
        test_list_y.append(1)
    elif(images_df_test[j][1]==0.5 and text_df_test[j][1]==0.5):
        continue
    else:
        test_list_y.append(0)    

    test_list_x.append([images_df_test[j][0],text_df_test[j][0]])
    

        
spam = 0
ham = 0
for i in range(4010):
    if train_list_y[i] == 1:
        spam+=1
    else:
        ham+=1
    


print(spam)
print(ham)


#lmodel,lhistory = activate_fusion_model(train_list_x,train_list_y,test_list_x,test_list_y,1)










    
    
    


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


#[accuracy,precision,recall,f1score,auc,kappa,confusionMatrix]  for each model/fold in 5fold
metrics_list = []


#exploring various metrics:
#returns 2d array



model = load_model(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\1st type of dataset - only images fusion models\fusion_model_fold1.h5")
model.summary()

yhat_probs = []
yhat_classes = []
for i in range(len(test_list_x)):
    
    kprobs = model.predict(np.array([test_list_x[i]]))
    yhat_probs.append(kprobs[0][0])
  
    if int(kprobs[0][0]) > 0.5:    
        yhat_classes.append(1)
    else:
        yhat_classes.append(0)
        
        
yhat_classes = []
for i in yhat_probs:
    
    if i>0.5:
        yhat_classes.append(1)
    else:
        yhat_classes.append(0)        


#2d array to be converted into 1d array for sklearn evaluations
yhat_probs = np.array(yhat_probs)
#yhat_probs = yhat_probs[:,0]

# from sklearn.metrics import classification_report
# report = classification_report(test_list_y, yhat_classes)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_list_y, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_list_y, yhat_classes,pos_label = 0)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_list_y, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_list_y, yhat_classes)
print('F1 score: %f' % f1)
# ROC AUC
auc = roc_auc_score(test_list_y, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(test_list_y, yhat_classes)
print(matrix)
# kappa
kappa = cohen_kappa_score(test_list_y, yhat_classes)
print('Cohens kappa: %f' % kappa)
print("fold4")

metrics_list.append([accuracy,precision,recall,f1,auc,kappa,matrix])



ytest = test_list_y
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
    
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat_probs)

# generate a no skill prediction (majority class)
ns_probs = [1 for _ in range(len(ytest))]

# predict probabilities
lr_probs = yhat_probs    
# keep probabilities for the positive outcome only
#lr_probs = lr_probs[:, 0]

# calculate scores
ns_auc = roc_auc_score(ytest, ns_probs)
lr_auc = roc_auc_score(ytest, lr_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)


# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate   (fold1)')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()






with open(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\1st type of dataset - only images fusion models\fusion_modelHistory_fold5","rb") as file_pi:
        vgg_classifier = pickle.load(file_pi)



print(vgg_classifier.keys())
# summarize history for accuracy
plt.plot(vgg_classifier['accuracy'])
plt.plot(vgg_classifier['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('binary_accuracy')
plt.xlabel('epoch   (fold5)')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(vgg_classifier['loss'])
plt.plot(vgg_classifier['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch    (fold5)')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





