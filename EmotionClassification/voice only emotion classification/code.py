
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')



#Load the Dataset
paths = []
labels = []
for dirname, _, filenames in os.walk(r"D:\dev\projects\IBM skills build emotion recognition multi model\voice emotion recognition\archive\TESS Toronto emotional speech set data"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')



len(paths)


paths[:5]

labels[:5]


## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


df['label'].value_counts()


#Exploratory Data Analysis
sns.countplot(df['label'])

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    
    
    
emotion = 'fear'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
print('\n')
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'angry'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'disgust'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'neutral'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'sad'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'ps'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'happy'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)





#Feature Extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

extract_mfcc(df['speech'][0])


X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))


X_mfcc

X = [x for x in X_mfcc]
X = np.array(X)
X.shape

## input split
X = np.expand_dims(X, -1)
X.shape

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()
y.shape







#Create the LSTM Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()




# Train the model
history = model.fit(X, y, validation_split=0.2, epochs=200, batch_size=64)



from tensorflow.keras.models import save_model
save_model(model, r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\model.h5")



#Plot the results

epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()




loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()




def final(df,model):
    


    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    X = [x for x in X_mfcc]
    X = np.array(X)
    X.shape
    X = np.expand_dims(X, -1)
    prediction = model.predict(X)
    prediction = np.argmax(prediction)
    
    if(prediction == 0):
            print('angry')
    if(prediction == 1):
            print('disgust')
    if(prediction == 2):
            print('fear')
    if(prediction == 3):
            print('happy')
    if(prediction == 4):
            print('neutral')
    if(prediction == 5):
            print('pleasent_surprise')
    if(prediction == 6):
            print('sad')
    
    
df1 = df.loc[[210],['speech']]
    
df2 = df.loc[[210],['speech','label']]

df3 = df.loc[[0],['speech']]



# X_mfcc = df1['speech'].apply(lambda x: extract_mfcc(x))
# X = [x for x in X_mfcc]
# X = np.array(X)
# X.shape
# X = np.expand_dims(X, -1)
# prediction = model.predict(X)
# prediction = np.argmax(prediction)

final(df1,model)
