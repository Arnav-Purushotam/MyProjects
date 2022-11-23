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


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from matplotlib.pyplot import plt



def create_fusion_model():
    model = Sequential()
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
    total_probabilities = np.concatenate((final_list_train,final_list_test),axis=0)
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    total_y = np.concatenate((y_pred,y_test),axis=0)
    history = model.fit(x = total_probabilities, validation_split = 0.2500843284, y = total_y, epochs = 32)
    link_to_save = r'D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\1st type of dataset - only images fusion models\fusion_model_fold' + str(fold) + '.h5'
    if os.path.isfile(link_to_save) is False:
        model.save(link_to_save)
    
    #saving history as a dict
    #history object holds different training metrics spanned accross every training epoch
    link_to_save_history = r'D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\1st type of dataset - only images fusion models\fusion_modelHistory_fold' + str(fold)
    with open(link_to_save_history,"wb") as file_pi:
        pickle.dump(history.history,file_pi)
        
    return link_to_save,link_to_save_history
    
    


def data_creator(model_path,data_path_train):
    
    fusion_list = []
    #new_cnn_model = load_model(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold1.h5")
    new_cnn_model = model_path
    #image_path = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold1\test\combined_test_spam_ham"
    image_path = data_path_train
    image_list = os.listdir(image_path)
    i = 0
    for test_img in image_list:
        if i%10 != 0:
            test_img = image.load_img(os.path.join(image_path,test_img),target_size=(224,224))
            test_img = image.img_to_array(test_img)
            test_img = np.array([test_img])
            test_img = test_img/255
            test_img = test_img.reshape(1,224,224,3)
            result = new_cnn_model.predict(test_img)
            #fusion_list contains the probability os spam
            fusion_list.append([result[0][1]])
            print(i)
        else:
            fusion_list.append([0.5])        
        i+= 1

    return fusion_list
        

listf = data_creator()
listf_test = data_creator()




def preprocessing(data):
      lem = WordNetLemmatizer()
      sms = contractions.fix(data) # converting shortened words to original (Eg:"I'm" to "I am")
      sms = sms.lower() # lower casing the sms
      sms = re.sub(r'https?://S+|www.S+', "", sms).strip() #removing url
      sms = re.sub("[^a-z ]", "", sms) # removing symbols and numbes
      sms = sms.split() #splitting
      # lemmatization and stopword removal
      sms = [lem.lemmatize(word) for word in sms if not word in set(stopwords.words("english"))]
      sms = " ".join(sms)
      return sms


def raw_text_data_generator(df):
# =============================================================================
#     df = pd.read_csv(r"D:\SPAM DATASET\datasets\text_dataset\spam_ham_dataset.csv", encoding='latin-1')
# =============================================================================
    df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1, inplace=True)
    #use axis = 1 when dropping columns
    df.columns = ["SpamHam","Tweet"]
    X = df["Tweet"].apply(preprocessing)
    with open('tokenizer.pickle', 'rb') as handle:
         tokenizer = pickle.load(handle)
    tokenizer.fit_on_texts(X)
    text_to_sequence = tokenizer.texts_to_sequences(X)
    max_length_sequence = 79
    padded_emails_input = pad_sequences(text_to_sequence, maxlen = max_length_sequence, padding = "post")
    la = LabelEncoder()
    y = la.fit_transform(df["SpamHam"])
    
    return padded_emails_input




def text_data_creator(num_entries):
    text_list = []
    new_lstm_model = load_model(r"D:\dev\projects\SPAM research\models\lstm_model.h5")
    df = pd.read_csv(r"D:\dev\projects\SPAM research\DATASETS\text_dataset\spam_fusion_dataset.csv", encoding='latin-1')
    #multiple_txt_input = raw_text_data_generator(df)
    listg = []
    for i in range(num_entries):
        #if i%25 != 0
        if 25 == 0:
            #my_input = multiple_txt_input[i].reshape(1,79)
            #result = new_lstm_model.predict(my_input)
            #listg.append([result[0][0]])
            print("hello")
        else:
            listg.append([0.5])
            
    return listg
        
        
        
listg = text_data_creator(14823)
listg_test = text_data_creator(3707)
    


def fusion_model_train_and_test_data_generator(listf,listg):
    random.shuffle(listf)
    random.shuffle(listg)
    listf = np.array(listf)
    listg = np.array(listg)
    final_list = np.concatenate((listf,listg), axis = 1)
    #listf is images and listg is text
    y = []
    for entry in final_list:
        if (entry[0] == 0.5 and entry[1] == 0.5):
            del entry
            
    for entry in final_list:
        if (entry[0] > 0.5 or entry[1] > 0.5):
            y.append(1)   #1 is to indicate spam
        else:
            y.append(0)   #0 is to indicate ham
    
    y = np.array(y)
    print(y.shape)
    return final_list,y
    

fold = 1
def fusion_model_trainer(listf,listg,listf_test,listg_test,fold):
    final_list,y_pred = fusion_model_train_and_test_data_generator(listf, listg)
    final_list_test,y_test = fusion_model_train_and_test_data_generator(listf_test, listg_test)
    i,j = activate_fusion_model(final_list, y_pred,final_list_test,y_test,fold)
    return i,j, final_list_test, y_test
    
    

# final_list,y_pred = fusion_model_train_and_test_data_generator(listf, listg)
# final_list_test,y_test = fusion_model_train_and_test_data_generator(listf_test, listg_test)



# activate_fusion_model(final_list,y_pred,final_list_test,y_test,fold)    



# def total_train_function_1st_dataset():
#     #this function will produce 5 different fusion models trained on 5 folds of image and corresponding 5 folds of text
    
#     model_list = []
#     total_metrics_list = []
#     for i in range(1,6):
        
#         image_model_path = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold" + str(i) + ".h5"
#         image_data_path_train = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold" + str(i) + "\train\combined spam ham train"
#         image_data_path_test = "D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold" + str(i) + "\test\combined_test_spam_ham"
        
#         listf = data_creator(image_model_path,image_data_path_train)
#         listf_test = data_creator(image_model_path,image_data_path_test)
        
#         #after creating individual text models from the text data folds, put the corresponding links here
#         text_model_path = r"" + str(i) + ".h5"
#         text_data_path_train = r"" 
#         text_data_path_text = r""
        
#         #change the text data creator according to ur dataset type and run this total function for all dataset types,
#         #where u are simply evalutaing a trained fusion model
#         #since initially u have to train the fusion models and save them, so change text data creator function
#         listg = text_data_creator(14823)
#         listg_test = text_data_creator(3707)
        
#         #train the fusion model here, after saving all the folds, only evaluate the models using the custom data, DONT TRAIN ON THEM
#         link_to_save,link_to_save_history,final_list_test, ytest = fusion_model_trainer(listf,listg,listf_test,listg_test,i)
        
        
        
#         my_reloaded_model = tf.keras.models.load_model((link_to_save),custom_objects={'KerasLayer':hub.KerasLayer})
#         model_list.append(my_reloaded_model)
        
        
#         #[accuracy,precision,recall,f1score,auc,kappa,confusionMatrix]  for each model/fold in 5fold
#         metrics_list = []


#         #exploring various metrics:
#         #returns 2d array




#         yhat_probs = []
#         yhat_classes = []
#         for j in range(len(final_list_test)):
#             kprobs = model_list[i].predict(final_list_test[j], verbose = 0)
#             kclasses = np.argmax(kprobs)
#             #kprobs[1] is probability of spam
#             yhat_probs.append(kprobs[1])
#             yhat_classes.append(kclasses)

#         #2d array to be converted into 1d array for sklearn evaluations
#         yhat_probs = np.array(yhat_probs)
#         yhat_probs = yhat_probs[:,0]
        
        
#         # accuracy: (tp + tn) / (p + n)
#         accuracy = accuracy_score(ytest, yhat_classes)
#         print('Accuracy: %f' % accuracy)
#         # precision tp / (tp + fp)
#         precision = precision_score(ytest, yhat_classes)
#         print('Precision: %f' % precision)
#         # recall: tp / (tp + fn)
#         recall = recall_score(ytest, yhat_classes)
#         print('Recall: %f' % recall)
#         # f1: 2 tp / (2 tp + fp + fn)
#         f1 = f1_score(ytest, yhat_classes)
#         print('F1 score: %f' % f1)
#         # ROC AUC
#         auc = roc_auc_score(ytest, yhat_probs)
#         print('ROC AUC: %f' % auc)
#         # confusion matrix
#         matrix = confusion_matrix(ytest, yhat_classes)
#         print(matrix)
#         # kappa
#         kappa = cohen_kappa_score(ytest, yhat_classes)
#         print('Cohens kappa: %f' % kappa)

#         metrics_list.append([accuracy,precision,recall,f1,auc,kappa,matrix])
#         total_metrics_list.append(metrics_list)
        
        
#         # calculate roc curve
#         fpr, tpr, thresholds = roc_curve(ytest, yhat_probs)
        
#         # generate a no skill prediction (majority class)
#         ns_probs = [0 for _ in range(len(ytest))]
        
#         # predict probabilities
#         lr_probs = yhat_probs    
        
        
#         # calculate scores
#         ns_auc = roc_auc_score(ytest, ns_probs)
#         lr_auc = roc_auc_score(ytest, lr_probs)
        
#         # calculate roc curves
#         ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)
#         lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)
        
        
#         # plot the roc curve for the model
#         plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#         plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
#         # axis labels
#         string = 'False Positive Rate' + str(i)
#         plt.xlabel(string)
#         plt.ylabel('True Positive Rate')
#         # show the legend
#         plt.legend()
#         # show the plot
#         plt.show()
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    

    
    
    
    
    

# final_list = np.concatenate((listf,listg), axis = 1)   














# =============================================================================
# def single_prediction(img_path,txt_path):    
# 
#     new_cnn_model = load_model(r'D:\SPAM DATASET\models\cnn_model.h5')
#     test_img = image.load_img(img_path,target_size=(128,128))
#     test_img = image.img_to_array(test_img)
#     test_img = np.array([test_img])
#     test_img = test_img/255
#     result1 = new_cnn_model.predict(test_img)
#     result_img = result1[0][0]
#     
#         #text prediction
# 
#     new_lstm_model = load_model(r'D:\SPAM DATASET\models\lstm_model.h5')
#     df1= pd.read_csv(txt_path, encoding='latin-1')
#     #df1 = df1.iloc[2:3,:]
#     multiple_txt_input = raw_text_data_generator(df1)
#     my_input = multiple_txt_input.reshape(1,79)
#     result_txt = new_lstm_model.predict(my_input)
#     
#     total_result = [[result_img,result_txt[0][0]]]
#     total_result = np.array(total_result)
#     
#     #fusion prediction
# 
#     new_fusion_model = load_model(r'D:\SPAM DATASET\models\fusion_model.h5')
#     result_final = new_fusion_model.predict(total_result)
#     if result_final[0][0] < 0.5:
#         print("input is NOT spam")
#     else:
#         print("input IS spam")
# 
# 
# =============================================================================


#result_final = single_prediction(r'D:\SPAM DATASET\single_prediction\email3\image_ham10445ki.jpg', r"D:\SPAM DATASET\single_prediction\email3\email3_text - Sheet1.csv")
    

        
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    


# def single_prediction(email_path):


#     path = email_path
#     folder_list = os.listdir(path)
    
#     i=0
#     for item in folder_list:
#         if i==1:
#             new_cnn_model = load_model(r'D:\SPAM DATASET\models\cnn_model.h5')
#             print(new_cnn_model.summary)
#             test_img = image.load_img(os.path.join(path,item),target_size=(128,128))
#             test_img = image.img_to_array(test_img)
#             test_img = np.array([test_img])
#             test_img = test_img/255
#             result1 = new_cnn_model.predict(test_img)
#             result_img = result1[0][0]
#         if i == 0:
#             #text prediction

#             new_lstm_model = load_model(r'D:\SPAM DATASET\models\lstm_model.h5')
#             print(new_lstm_model.summary)
#             df1= pd.read_csv(os.path.join(path,item), encoding='latin-1')
#             #df1 = df1.iloc[2:3,:]
#             multiple_txt_input = raw_text_data_generator(df1)
#             my_input = multiple_txt_input.reshape(1,79)
#             result_txt = new_lstm_model.predict(my_input)
            
        
          
#         i+=1
    
    

    
#     total_result = [[result_img,result_txt[0][0]]]
#     total_result = np.array(total_result)
#     print('\n')
#     print('\n')
#     print("first is image spam classification prob and 2nd is text spam classification probability : {}".format(total_result))
#     print('\n')
#     print('\n')
#     #fusion prediction

#     new_fusion_model = load_model(r'D:\SPAM DATASET\models\fusion_model.h5')
#     print("new_fusion_model.summary")
#     result_final = new_fusion_model.predict(total_result)
#     print('\n')
#     print('\n')
#     print("the final probability value is {}".format(result_final))
#     print('\n')
#     print('\n')
#     if result_final[0][0] < 0.5:
#         return "input is NOT spam"
#     else:
#         return "input IS spam"

    
# from tkinter import *
# from functools import partial  
    
  
# # Top level window
# frame = Tk()
# frame.title("Email Spam Detection")
# frame.geometry('1350x750+0+0')
# frame.configure(background = 'light blue')
# Tops = Frame(frame,bg='light blue',bd = 20,pady=5,relief = RIDGE)
# Tops.pack(side=TOP)
# lblTitle = Label(Tops,font=('arial',60,'bold'),text='Email Spam Detection',bd=21,bg='black',fg='cornsilk',justify=CENTER)
# lblTitle.grid(row=0)
# # Function for getting Input
# # from textbox and printing it 
# # at label widget
  
# def printInput():
#     inp= inputtxt1.get(1.0, "end-1c")
    
#     todisplay = single_prediction(inp)  
    
#     lbl.config(text = "Provided Input: "+todisplay)


# lbl1 = Label(frame,font=('arial',50,'bold'), text = "Enter Folder Location Below: ",padx=30,pady=2,bg='blue')
# lbl1.pack( fill = X)
# # TextBox Creation
# inputtxt1 = Text(frame,font=('arial',20,'bold'),
#                     height = 1,
#                     width = 100)


# inputtxt1.pack(fill=X)

  
# # Button Creation
# printButton = Button(frame,font=('arial',20,'bold'),
#                         text = "Check", command = printInput, pady=3, padx=50,bd=7)
# printButton.pack(pady = 10)
  
# # Label Creation
# lbl2 = Label(frame,font=('arial',50,'bold'), text = "Your Result: ",padx=600,pady=2,bg='blue')
# lbl2.pack(fill=X)
# lbl = Label(frame,font=('arial',20,'bold'), text = "",padx=20)
# lbl.pack( fill =X)
# frame.mainloop()

# =============================================================================
#   
# # Top level window
# frame = tk.Tk()
# frame.title("Email Spam Detection")
# frame.geometry('1350x750+0+0')
# frame.configure(background = 'light blue')
# Tops = Frame(frame,bg='light blue',bd = 20,pady=5,relief = RIDGE)
# Tops.pack(side=TOP)
# lblTitle = Label(Tops,font=('arial',60,'bold'),text='Email Spam Detection',bd=21,bg='black',fg='cornsilk',justify=CENTER)
# lblTitle.grid(row=0)
# # Function for getting Input
# # from textbox and printing it 
# # at label widget
#   
# def printInput():
#     inp= inputtxt1.get(1.0, "end-1c")
#     
#     todisplay = single_prediction(inp)  
#     
#     lbl.config(text = "Provided Input: "+todisplay)
# 
# 
# lbl1 = tk.Label(frame,font=('arial',50,'bold'), text = "Enter Folder Location Below: ",padx=30,pady=2,bg='yellow')
# lbl1.pack( fill = X)
# # TextBox Creation
# inputtxt1 = tk.Text(frame,
#                     height = 2,
#                     width = 100)
# 
# 
# inputtxt1.pack(fill=X)
# 
#   
# # Button Creation
# printButton = tk.Button(frame,font=('arial',20,'bold'),
#                         text = "Check", 
#                         command = printInput, pady=20, padx=50,bd=7)
# printButton.pack()
#   
# # Label Creation
# lbl2 = tk.Label(frame,font=('arial',50,'bold'), text = "Your Result: ",padx=600,pady=2,bg='yellow')
# lbl2.pack(fill=X)
# lbl = tk.Label(frame, text = "",padx=20,pady=10)
# lbl.pack( fill =X)
# frame.mainloop()
# =============================================================================








