#building the LSTM model

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import collections
import contractions
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from nltk.stem.wordnet import WordNetLemmatizer as lem
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('omw-1.4')
# =============================================================================
# from tf.keras.layers import Dense, Embedding, LSTM, Dropout
# from tf.keras.models import Sequential

# 
# from tf.keras.models import load_model
# encoding='latin-1'
# =============================================================================



df_train = pd.read_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold1\fold1_train.csv")
df_test = pd.read_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold1\fold1_test.csv" )
df =pd.concat([df_train,df_test], ignore_index=True)
df = df.iloc[:,:2]



df.columns = ["SpamHam","Tweet"]
df.Tweet=df.Tweet.astype(str)



def preprocessing(data):
     
      sms = contractions.fix(data) # converting shortened words to original (Eg:"I'm" to "I am")
      sms = sms.lower() # lower casing the sms
      sms = re.sub(r'https?://S+|www.S+', "", sms).strip() #removing url
      sms = re.sub("[^a-z ]", "", sms) # removing symbols and numbes
      sms = sms.split() #splitting
      # lemmatization and stopword removal
      sms = [lem.lemmatize(word) for word in sms if not word in set(stopwords.words("english"))]
      sms = " ".join(sms)
      return sms
X = df["Tweet"].apply(preprocessing)



tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
text_to_sequence = tokenizer.texts_to_sequences(X)

max_length_sequence = max([len(i) for i in text_to_sequence])
padded_emails_input = pad_sequences(text_to_sequence, maxlen = max_length_sequence, padding = "post")


from sklearn.preprocessing import LabelEncoder

la = LabelEncoder()
y = la.fit_transform(df["SpamHam"])

del  lem, stopwords, text_to_sequence, X


totalsize = len(tokenizer.word_index) + 1





def create_model():
    max_features =7836
    embedding_dim =16
    sequence_length = max_length_sequence

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features +1, embedding_dim, input_length=sequence_length,\
                                        embeddings_regularizer = regularizers.l2(0.005))) 
    #model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.LSTM(embedding_dim,dropout=0.2, recurrent_dropout=0.2,return_sequences=True,\
                                                                 kernel_regularizer=regularizers.l2(0.005),\
                                                                 bias_regularizer=regularizers.l2(0.005)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, activation='relu',\
                                kernel_regularizer=regularizers.l2(0.001),\
                                bias_regularizer=regularizers.l2(0.001),))
    #model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Dense(8, activation='relu',\
                                kernel_regularizer=regularizers.l2(0.001),\
                                bias_regularizer=regularizers.l2(0.001),))    
    #model.add(tf.keras.layers.Dropout(0.4))


    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
                               



    model.summary()
    
    from keras.optimizers import SGD
    opt = SGD(lr = 0.01)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(1e-3),metrics=[tf.keras.metrics.BinaryAccuracy()])


    return model

lstm_model = create_model()


#None in model summary is the batch size


history = lstm_model.fit(padded_emails_input,epochs = 20,validation_split = 0.2,batch_size = 10,y = y)






import pickle
import os



if os.path.isfile(r'D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\models\fold1.h5') is False:
    lstm_model.save(r'D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\models\fold1.h5')

#saving history as a dict
#history object holds different training metrics spanned accross every training epoch
with open(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\models\Fold1historyDict","wb") as file_pi:
    pickle.dump(history.history,file_pi)






    

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



ytest = df_test.iloc[:,0]
Xtest = df_test.iloc[:,1]
Xtest = Xtest.to_frame().reset_index()
Xtest.rename(columns = {"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...":"Tweet"}, inplace = True)
Xtest.Tweet=Xtest.Tweet.astype(str)




Xtest = Xtest["Tweet"].apply(preprocessing)
from nltk.corpus import stopwords
lem = nltk.stem.wordnet.WordNetLemmatizer()


tokenizer = Tokenizer()
tokenizer.fit_on_texts(Xtest)
text_to_sequence = tokenizer.texts_to_sequences(Xtest)

max_length_sequence = max([len(i) for i in text_to_sequence])
max_length_sequence = 70
padded_emails_input = pad_sequences(text_to_sequence, maxlen = max_length_sequence, padding = "post")


from sklearn.preprocessing import LabelEncoder

la = LabelEncoder()
y = la.fit_transform(df["SpamHam"])




totalsize = len(tokenizer.word_index) + 1


import tensorflow_hub as hub
from tensorflow.keras.models import load_model
new_lstm_model = load_model(r'D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\models\fold1.h5')
new_lstm_model.summary()

yhat_probs = []
yhat_classes = []
for j in range(len(padded_emails_input)):
    my_input = padded_emails_input[j].reshape(1,70)
    kprobs = new_lstm_model.predict(my_input)
    
    yhat_probs.append(kprobs[0])
    if kprobs[0][0] > 0.5:    
        yhat_classes.append('spam')
    else:
        yhat_classes.append('ham')
        

#2d array to be converted into 1d array for sklearn evaluations
yhat_probs = np.array(yhat_probs)
yhat_probs = yhat_probs[:,0]


ytest = ytest.to_numpy()

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes,pos_label = 'ham')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes,pos_label = 'ham')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes,pos_label = 'ham')
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







































   


print(padded_emails_input[:1].shape)



import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()    

# =============================================================================
# # loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
# =============================================================================

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#print(classification_report(, history.history['val_binary_accuracy']))






    
    

