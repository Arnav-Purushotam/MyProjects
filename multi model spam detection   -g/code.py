
#building the CNN model

#--------------------------------------------------------------------------------------
import io
import requests
from PIL import Image
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.metrics
import tensorflow.keras.losses
from PIL import Image, ImageFile
#import tensorflow_hub as hub
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pickle
import os

#--------------------------------------------------------------------------------------


#pre processing training set
train_datagen = ImageDataGenerator(rescale = 1./255)
                                    #horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
training_set = train_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\ad images bulk google image downloads\TRAIN",
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'binary')



#pre processing the test set
test_datagen = ImageDataGenerator(rescale = 1./255,)#horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
test_set = test_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\ad images bulk google image downloads\TEST",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')


#-------------------------------------------------------------------------------------
#building the cnn






# def create_model():
#     cnn = Sequential()
#     cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=[256, 256, 3]))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=2, strides=2))
#     cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=[128, 128, 3]))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=2, strides=2))
#     cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=[64, 64, 3]))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=2, strides=2))
#     cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=[32, 32, 3]))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(Flatten())
#     cnn.add(Dense(128))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(Dense(64))
#     cnn.add(Activation('relu'))
#     cnn.add(Dense(32))
#     cnn.add(Activation('relu'))
#     cnn.add(Dense(1,activation='sigmoid'))
    
#     return cnn



# def create_model():
#     cnn = Sequential()
#     cnn.add(Conv2D(filters=96, kernel_size=3, input_shape=[128, 128, 3],strides=(1,1)))
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=3, strides=2))
#     cnn.add(Conv2D(filters=128, kernel_size=3, input_shape=[96, 62, 62],strides=(1,1)))
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=3, strides=2))
#     cnn.add(Flatten())
#     cnn.add(Dense(256))
#     cnn.add(Activation('relu'))
#     cnn.add(Dropout(0.1))
#     cnn.add(Dense(1,activation='sigmoid'))
#     return cnn




# def create_model():
    
#     feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
#     pretrained_model_without_top_layer = hub.KerasLayer(
#     feature_extractor_model, input_shape=[224, 224, 3], trainable=False)
#     model = tf.keras.Sequential([
#     pretrained_model_without_top_layer,
#     tf.keras.layers.Dense(1,activation = 'softmax')
# ])
#     model.summary()
#     return model
  



# def create_model():
#     cnn = Sequential()
#     cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=[128, 128, 3]))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=2, strides=2))
#     cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=[64, 64, 3]))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=2, strides=2))
#     cnn.add(Conv2D(filters=32, kernel_size=3, input_shape=[32, 32, 3]))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(MaxPool2D(pool_size=2, strides=2))
#     cnn.add(Flatten())
#     cnn.add(Dense(64))
#     cnn.add(BatchNormalization())
#     cnn.add(Activation('relu'))
#     cnn.add(Dense(32))
#     cnn.add(Activation('relu'))
#     cnn.add(Dense(1,activation='softmax'))

#     return cnn

#-------------------------------------------------------------------------------------




# model = create_model()
# #model.compile(optimizer="adam",
# #              loss=tf.keras.losses.BinaryCrossentropy(),
# #             metrics=tf.keras.metrics.Accuracy())
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(x = training_set, validation_data = test_set, epochs = 30)               


#-------------------------------------------------------------------------------------
# #to load pre trained model
# import tensorflow_hub as hub
# # from tensorflow.keras.models import load_model
# my_reloaded_model = tf.keras.models.load_model((r'D:\SPAM DATASET\cnn_mobilenet_success_AUGDATASET1.h5'),custom_objects={'KerasLayer':hub.KerasLayer})





# import os
# total_images = []



# path = r"D:\SPAM DATASET\new_dataset\train\train_ham"
# folder_list = os.listdir(path)

# for item in folder_list:
#     filename = os.path.join(path,item)
#     #adding class when loading images ham: 0, spam : 1
#     k = [filename,0]
#     total_images.append(k)
    
    
    
#   #converting to dataframe so we can shuffle the images   
# total_images = pd.DataFrame(total_images, columns = ['Name', 'class'])  
#  # shuffle the DataFrame rows
# total_images = total_images.sample(frac = 1)

# #seperating the X and y for train and test sets

# X = total_images.iloc[:,0]
# y = total_images.iloc[:,1]
# X = X.to_numpy()
# y = y.to_numpy()

# #manual preprocessing and loading of images, without using datagens
# from tensorflow.keras.preprocessing import image
# for i in range(len(X)):
#     try:
        
#         test_img = image.load_img(X[i],target_size=(224,224))
#         test_img = image.img_to_array(test_img)
#         test_img = np.array([test_img])
#         test_img = test_img/255
#         X[i] = test_img
#     except:
#         print("no")
    
    


# #reshaping data in order to be fed as input to the model
# X_train[0] = X_train[0].reshape(1,224,224,3)
# X_train = X_train.to_numpy()


# X_train = X_train.reshape(X_train.shape[0],224,224,3)
# my_reloaded_model.predict(X_train,y_train)
#-------------------------------------------------------------------------------------





# #saving the model
# import os
# if os.path.isfile(r'D:\SPAM DATASET\models\cnn_model_successjh.h5') is False:
#     model.save(r'D:\SPAM DATASET\models\cnn_mobilenet_new.h5')
 
#-------------------------------------------------------------------------------------


# #plotting the model results
 
# import matplotlib.pyplot as plt
# import numpy

# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['binary_accuracy'])
# plt.plot(history.history['val_binary_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('binary_accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# new_cnn_model.summary()


#-------------------------------------------------------------------------------------



def model_maker():

    IMG_SHAPE  = 224
    batch_size = 32

    pre_trained_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers:
        print(layer.name)
        layer.trainable = False
    

    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output
    x = tf.keras.layers.GlobalMaxPooling2D()(last_output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation='sigmoid')(x)
    model = tf.keras.Model(pre_trained_model.input, x)
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])



    # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    # mc = tf.keras.callbacks.ModelCheckpoint(r'D:\dev\projects\SPAM research\models\early stopping vggnet\best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


    # vgg_classifier = model.fit(training_set,
    #                            epochs = 10,
    #                            validation_data=test_set,
    #                            batch_size = batch_size,
    #                            verbose = 1,callbacks=[es, mc])
    
    return model











import os
history_list = []
model_list = []





model = model_maker()
model.summary()





train_datagen = ImageDataGenerator(rescale = 1./255)
                                    #horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
training_set = train_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold1\train",
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'binary')



#pre processing the test set
test_datagen = ImageDataGenerator(rescale = 1./255,)#horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
test_set = test_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold1\test",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = model_maker()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = tf.keras.callbacks.ModelCheckpoint(r'D:\dev\projects\SPAM research\models\early stopping vggnet\best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


history = model.fit(training_set,
                            epochs = 20,
                            validation_data=test_set,
                            batch_size = 32,
                            verbose = 1,callbacks=[es, mc])

history_list.append(history)
model_list.append(model)










train_datagen = ImageDataGenerator(rescale = 1./255)
                                    #horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
training_set = train_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold2\train",
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'binary')



#pre processing the test set
test_datagen = ImageDataGenerator(rescale = 1./255,)#horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
test_set = test_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold2\test",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = model_maker()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = tf.keras.callbacks.ModelCheckpoint(r'D:\dev\projects\SPAM research\models\early stopping vggnet\best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


history = model.fit(training_set,
                            epochs = 20,
                            validation_data=test_set,
                            batch_size = 32,
                            verbose = 1,callbacks=[es, mc])

history_list.append(history)
model_list.append(model)











train_datagen = ImageDataGenerator(rescale = 1./255)
                                    #horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
training_set = train_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold3\train",
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'binary')



#pre processing the test set
test_datagen = ImageDataGenerator(rescale = 1./255,)#horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
test_set = test_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold3\test",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = model_maker()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = tf.keras.callbacks.ModelCheckpoint(r'D:\dev\projects\SPAM research\models\early stopping vggnet\best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


history = model.fit(training_set,
                            epochs = 20,
                            validation_data=test_set,
                            batch_size = 32,
                            verbose = 1,callbacks=[es, mc])

history_list.append(history)
model_list.append(model)










train_datagen = ImageDataGenerator(rescale = 1./255)
                                    #horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
training_set = train_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold4\train",
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'binary')



#pre processing the test set
test_datagen = ImageDataGenerator(rescale = 1./255,)#horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
test_set = test_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold4\test",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = model_maker()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = tf.keras.callbacks.ModelCheckpoint(r'D:\dev\projects\SPAM research\models\early stopping vggnet\best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


history = model.fit(training_set,
                            epochs = 20,
                            validation_data=test_set,
                            batch_size = 32,
                            verbose = 1,callbacks=[es, mc])

history_list.append(history)
model_list.append(model)












train_datagen = ImageDataGenerator(rescale = 1./255)
                                    #horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
training_set = train_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold5\train",
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'binary')



#pre processing the test set
test_datagen = ImageDataGenerator(rescale = 1./255,)#horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1.5],rotation_range=90)
test_set = test_datagen.flow_from_directory(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold5\test",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = model_maker()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = tf.keras.callbacks.ModelCheckpoint(r'D:\dev\projects\SPAM research\models\early stopping vggnet\best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


history = model.fit(training_set,
                            epochs = 20,
                            validation_data=test_set,
                            batch_size = 32,
                            verbose = 1,callbacks=[es, mc])

history_list.append(history)
model_list.append(model)






import pickle



if os.path.isfile(r'D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold5.h5') is False:
    model_list[4].save(r'D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold5.h5')

#saving history as a dict
#history object holds different training metrics spanned accross every training epoch
with open(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\model_histories\Fold5historyDict","wb") as file_pi:
    pickle.dump(history_list[4].history,file_pi)


    
    
# # list all data in history
# print(history.history.keys())
# output : 
# ['accuracy', 'loss', 'val_accuracy', 'val_loss']
    
    
    







total_images = []



path = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold5\test\ham"
folder_list = os.listdir(path)

for item in folder_list:
    filename = os.path.join(path,item)
    #adding class when loading images ham: 0, spam : 1
    k = [filename,0]
    total_images.append(k)
    

path1 = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold5\test\spam"
folder_list1 = os.listdir(path1)
for item in folder_list1:
    filename = os.path.join(path,item)
    #adding class when loading images ham: 0, spam : 1
    k = [filename,1]
    total_images.append(k)   
 
    
  #converting to dataframe so we can shuffle the images   
total_images = pd.DataFrame(total_images, columns = ['Name', 'class'])  
  # shuffle the DataFrame rows
total_images = total_images.sample(frac = 1)

#seperating the X and y

Xtest = total_images.iloc[:,0]
ytest = total_images.iloc[:,1]
Xtest = Xtest.to_numpy()
ytest = ytest.to_numpy()

#manual preprocessing and loading of images, without using datagens
from tensorflow.keras.preprocessing import image
for i in range(len(Xtest)):
    
        
        test_img = image.load_img(Xtest[i],target_size=(224,224))
        test_img = image.img_to_array(test_img)
        test_img = np.array([test_img])
        test_img = test_img/255
        Xtest[i] = test_img
   
    
    

Xtest = np.array(Xtest)


for i in range(len(Xtest)):
    Xtest[i] = Xtest[i].reshape(1,224,224,3)






model_list = []
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
my_reloaded_model = tf.keras.models.load_model((r'D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\models\fold5.h5'),custom_objects={'KerasLayer':hub.KerasLayer})
model_list.append(my_reloaded_model)





    
    

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




yhat_probs = []
yhat_classes = []
for j in range(len(Xtest)):
    kprobs = model_list[0].predict(Xtest[j], verbose = 0)
    kclasses = np.argmax(kprobs)
    yhat_probs.append(kprobs[1])
    yhat_classes.append(kclasses)

#2d array to be converted into 1d array for sklearn evaluations
yhat_probs = np.array(yhat_probs)
yhat_probs = yhat_probs[:,0]




# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes)
print('F1 score: %f' % f1)
# ROC AUC
auc = roc_auc_score(ytest, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(ytest, yhat_classes)
print(matrix)
# kappa
kappa = cohen_kappa_score(ytest, yhat_classes)
print('Cohens kappa: %f' % kappa)

metrics_list.append([accuracy,precision,recall,f1,auc,kappa,matrix])










from sklearn.metrics import roc_curve
from matplotlib.pyplot import plt
    
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat_probs)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(ytest))]

# predict probabilities
lr_probs = yhat_probs    
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 0]

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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
    









#to load back model history:
    # history = pickle.load(open(link,"rb"))




# total_images = []


# path = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold" + str(1) + r"\test\ham"
# folder_list = os.listdir(path)

# for item in folder_list:
#         filename = os.path.join(path,item)
#         #adding class when loading images ham: 0, spam : 1
#         k = [filename,0]
#         total_images.append(k)






metrics_list = []
for t in range(1,6):
    
    
    
    

    total_images = []



    path = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold" + str(t) + r"\test\ham"
    folder_list = os.listdir(path)

    for item in folder_list:
        filename = os.path.join(path,item)
        #adding class when loading images ham: 0, spam : 1
        k = [filename,0]
        total_images.append(k)
    

    path1 = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold" + str(t) + r"\test\spam"
    folder_list1 = os.listdir(path1)
    for item in folder_list1:
        filename = os.path.join(path1,item)
        #adding class when loading images ham: 0, spam : 1
        k = [filename,1]
        total_images.append(k)   
 
    
    #converting to dataframe so we can shuffle the images   
    total_images = pd.DataFrame(total_images, columns = ['Name', 'class'])  
    # shuffle the DataFrame rows
    total_images = total_images.sample(frac = 1)

    #seperating the X and y

    Xtest = total_images.iloc[:,0]
    ytest = total_images.iloc[:,1]
    Xtest = Xtest.to_numpy()
    ytest = ytest.to_numpy()

    #manual preprocessing and loading of images, without using datagens
    #from tensorflow.keras.preprocessing import image
    for i in range(len(Xtest)):
        try:
        
            test_img = image.load_img(Xtest[i],target_size=(224,224))
            test_img = image.img_to_array(test_img)
            test_img = np.array([test_img])
            test_img = test_img/255
            Xtest[i] = test_img
        except:
            print("no")
    
    

    Xtest = np.array(Xtest)


    for i in range(len(Xtest)):
        Xtest[i] = Xtest[i].reshape(1,224,224,3)
        
    
    
    

    yhat_probs = []
    yhat_classes = []
    for j in range(len(Xtest)):
        kprobs = model_list[t-1].predict(Xtest[j], verbose = 0)
        kclasses = np.argmax(kprobs)
        yhat_probs.append(kprobs[1])
        yhat_classes.append(kclasses)

    #2d array to be converted into 1d array for sklearn evaluations
    yhat_probs = np.array(yhat_probs)
    yhat_probs = yhat_probs[:,0]




    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(ytest, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(ytest, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(ytest, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(ytest, yhat_classes)
    print('F1 score: %f' % f1)
    # ROC AUC
    auc = roc_auc_score(ytest, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(ytest, yhat_classes)
    print(matrix)
    # kappa
    kappa = cohen_kappa_score(ytest, yhat_classes)
    print('Cohens kappa: %f' % kappa)

    metrics_list.append([accuracy,precision,recall,f1,auc,kappa,matrix])
        
        
    
    



metrics_list = np.array(metrics_list)
accuracy_total = np.mean(metrics_list[:,0])
print('Accuracy_total: %f' % accuracy_total)
precision_total = np.mean(metrics_list[:,1])
print('precision_total: %f' % precision_total)
recall_total = np.mean(metrics_list[:,2])
print('recall_total: %f' % recall_total)
f1score_total = np.mean(metrics_list[:,3])
print('f1score_total: %f' % f1score_total)
auc_total = np.mean(metrics_list[:,4])
print('auc_total: %f' % auc_total)
kappa_total = np.mean(metrics_list[:,5])
print('kappa_total: %f' % kappa_total)
    


    
    
    
    
    
    
    
#positive outcome here is no spam
# test_img = image.load_img(r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data\fold1\test\combined_test_spam_ham\NOSPAM_114.jpg",target_size=(224,224))
# test_img = image.img_to_array(test_img)
# test_img = np.array([test_img])
# test_img = test_img/255
# test_img = test_img.reshape(1,224,224,3)
# kprobs = my_reloaded_model.predict(test_img, verbose = 0)







# #plotting the model results
 
# import matplotlib.pyplot as plt
# import numpy

# print(vgg_classifier.history.keys())
# # summarize history for accuracy
# plt.plot(vgg_classifier.history['accuracy'])
# plt.plot(vgg_classifier.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('binary_accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(vgg_classifier.history['loss'])
# plt.plot(vgg_classifier.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()











