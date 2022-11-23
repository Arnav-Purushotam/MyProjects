from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'D:\dev\projects\IBM skills build emotion recognition multi model\face emotion recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier =load_model(r"D:\dev\projects\IBM skills build emotion recognition multi model\face emotion recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(r"D:\dev\projects\IBM skills build emotion recognition multi model\final video.mp4")



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








#################################################################################################################
# Load Yolo
net = cv2.dnn.readNet(r"D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\arabic video files\files-20220720T172825Z-001\files\yolov3.weights", r"D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\arabic video files\files-20220720T172825Z-001\files\yolov3.cfg")
classes = []
#loading the yolo classes into above list
#.strip() Removes spaces at the beginning and at the end of the string
#as you want part of the line only having the letters, not the rest of the blank space
with open(r"D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\arabic video files\files-20220720T172825Z-001\files\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
#net.getLayerNames(): It gives you list of all layers names used in a network. Like I am currently working with yolov3. It gives me a list of 254 layers.
#net.getUnconnectedOutLayers(): It gives you the final layers number in the list from net.getLayerNames()
layer_names = net.getLayerNames()
#to get the names of the actual output layers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]



def get_players(outs,height, width):
    class_ids = []
    confidences = []
    boxes = []
    players=[]
    sports_ball = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            #this is done as those scores are in a different format
            #it gets the index(in coco names) of the highest detected class
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                #for one detected object boxes list contains the co-ordinates
                #confidences list contains the confidence of the detected object's detection
                #class_ids list contains the index(in coco names) of the detected class
                #as the two forloops above run, this is done for all detected objects in the frame
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    #NMS boxes is a list of box objects, used to group together all the three above lists
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='person':
                players.append(boxes[i])
            if label=='sports ball':
                sports_ball.append(boxes[i])
            
    #returning just the co-ordinates now
    return players,sports_ball



#####################################################################################################################








#Feature Extraction
def extract_mfcc(filename):
    try:
        
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except:
        print("n")

        
        


def final(df,model):
    try:
        


        X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
        X = [x for x in X_mfcc]
        X = np.array(X)
        X.shape
        X = np.expand_dims(X, -1)
        prediction = model.predict(X)
        prediction = np.argmax(prediction)
    
        if(prediction == 0):
                prediction = 'angry'
        if(prediction == 1):
                prediction = 'disgust'
        if(prediction == 2):
                prediction = 'fear'
        if(prediction == 3):
                prediction = 'happy'
        if(prediction == 4):
                prediction = 'neutral'
        if(prediction == 5):
                prediction = 'pleasant_surprise'
        if(prediction == 6):
                prediction = 'sad'
            
        return prediction
    
    except:
        print("n")
    
    



total_list_instance = []
while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #faces = face_classifier.detectMultiScale(gray)

    #for (x,y,w,h) in faces:
     #   cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
      #  roi_gray = gray[y:y+h,x:x+w]
       # roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
       
    height, width, channels = frame.shape
        
    #creating a blob from the grayscaled frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
    #blob is created as the yolo dnn only accepts blob as input
    net.setInput(blob)
        
    # Runs a forward pass to compute the net output, input already set above
    # will give Numpy ndarray as output which you can use it to plot box on the given input image.
    outs = net.forward(output_layers)
    #executing a function to extract rectangle co-ordinates from obtained detections
    outs,outs_ball=get_players(outs, height, width)
    #iterate through all the detected components in a frame
    for i in range(len(outs)):
        #getting the co-ordinates of detected components(outs contains co-ordinates of detected components in that frame)
        x, y, w, h = outs[i]
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            total_list_instance.append(label)
            
            
            
            
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


chunks = [total_list_instance[x:x+24] for x in range(0, len(total_list_instance), 24)]



keymax_list = []
for list_instance in chunks:
    instance_dict = {"Neutral" : 0,
                     "Surprise" : 0,
                     "Sad" : 0,
                     "Happy" : 0,
                     "Disgust" : 0,
                     "Fear" : 0,
                     "Angry" : 0}
    for i in range(len(list_instance)):
        instance_dict[list_instance[i]]+=1

    Keymax = max(zip(instance_dict.values(), instance_dict.keys()))[1]
    keymax_list.append(Keymax)
    
    
    
    
    
    
    
 
    
    
    
    
    
########################################################################################################
    
    
#extracting audio from video  
import moviepy.editor as mp
#from moviepy.editor import *
my_clip = mp.VideoFileClip(r"D:\dev\projects\IBM skills build emotion recognition multi model\face emotion recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\Make anyone angry in 10.38 seconds.mp4")
my_clip.audio.write_audiofile(r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\my_result.mp3")
#audioclip = AudioFileClip("geeks.mp4")



#splitting audio into 1sec clips
from pydub import AudioSegment

audio = AudioSegment.from_file(r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\my_result.mp3")
lengthaudio = len(audio)
print("Length of Audio File", lengthaudio)

start = 0
# # In Milliseconds, this will cut 10 Sec of audio
threshold = 1000
end = 0
counter = 0

while start < len(audio):

    end += threshold

    print(start , end)

    chunk = audio[start:end]

    filename = r'D:\dev\projects\IBM skills build emotion recognition multi model\audio\1sec audio clips\audio clips\chunk' + str(counter) + '.wav'

    chunk.export(filename, format="wav")

    counter +=1

    start += threshold



########################################################################################################




#Load the Dataset
paths = []
labels = []
for dirname, _, filenames in os.walk(r"D:\dev\projects\IBM skills build emotion recognition multi model\voice emotion recognition\archive\final"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename
        labels.append(label.lower())
  
print('Dataset is Loaded')


## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
    
 
    
n = len(paths)

sound_prediction_list = []
for i in range(n):
    df1 = df.loc[[i],['speech']]
    model = load_model(r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\model.h5")
    prediction_sound = final(df1,model)
    sound_prediction_list.append(prediction_sound)
    

for i in range(len(sound_prediction_list)):
    print("\n")
    print(i)
    k = "audio : " + str(sound_prediction_list[i]) 
    print(k)
    
    try:
        k1 = "video : " + str(keymax_list[i]) 
        print(k1)       
    except:
        print("n")






































































        #making a copy of the frame
        copy=frame.copy()
        #converting to greyscale, the original frame
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

            
        
        

