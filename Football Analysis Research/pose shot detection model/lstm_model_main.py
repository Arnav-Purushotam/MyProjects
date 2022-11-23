import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
#from scipy import stats
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
#import seaborn as sns
#from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist



# columns = ['user','activity','timestamp','x-axis','y-axis','z-axis']
# df = pd.read_csv(r"D:\dev\projects\football AR\code\pose shot detection model\LSTM model datasets\WISDM_ar_latest\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt", header = None, names = columns)
# df = df.dropna()
# df.head()


# df.info()


# countOfActivity = df['activity'].value_counts()
# countOfActivity.plot(kind = 'bar', title = 'Training examples by activity type',figsize = (14,8))


# N_TIME_STEPS = 200
# N_FEATURES = 3
# step = 20
# segments = []
# labels = []
# for i in range(0, len(df) - N_TIME_STEPS, step):  #will give starting point of each batch
#     xs = df['x-axis'].values[i: i + N_TIME_STEPS]
#     ys = df['y-axis'].values[i: i + N_TIME_STEPS]
#     zs = df['z-axis'].values[i: i + N_TIME_STEPS]
#     label = stats.mode(df['activity'][i: i + N_TIME_STEPS]) #returns two arrays mode and count
#     label = label[0][0]
#     segments.append([xs, ys, zs])
#     labels.append(label)
    
    
# np.array(segments).shape

# reshaped_segments = np.asarray(segments).reshape(-1, N_TIME_STEPS, N_FEATURES)
# reshaped_segments.shape

# reshaped_segments = reshaped_segments.tolist()

# for entry in reshaped_segments:
    
#     for row in entry:
        
#         for value in row:
            
#               if isinstance(value, str):
#                   new_value = ""
#                   for k in value:
#                       if k != ';':
#                           new_value = new_value + k
#                   value = new_value
#                   value = float(value)
#               else:
#                 value = float(value)

# new_list = []
# for entry in reshaped_segments:
    
#     for row in entry:
        
#         for value in row:
            
#               if value != float:
#                   value = float(value)
            
                 




#one hot encoding of labels
#do label encoding if only 2 classes present

# labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

    

# X_train, X_test, y_train, y_test = train_test_split( reshaped_segments, labels, test_size=0.2)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#normalizing the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0




print(X_train.shape[1:])
#input_shape = (time_steps,no_of_columns_in_X_train)

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))
    




#Compiling the network
model.compile( loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'] )


#Fitting the data to the model
model.fit(X_train,
         y_train,
          epochs=20,
          validation_data=(X_test, y_test))


  

    





















    
    
    
    
    