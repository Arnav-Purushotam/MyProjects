import mediapipe as mp
import math
import cv2
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import os


#initializing mediapipe pose class
mp_pose = mp.solutions.pose

#settign up the pose functions
pose = mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.3, model_complexity = 1)

#initializing mediapipe drawing class, useful for annotation
mp_drawing = mp.solutions.drawing_utils



#reading an image and displaying using matplotlib
sample_img = cv2.imread(r"C:\Users\Asus\Desktop\football AR\32.jpg")
plt.figure(figsize = [10,10])
#display the sample image, also convert BGR to RGB for display
plt.title("sample image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()



#opencv also accepts in BGR while mediapipe in RGB
results = pose.process(cv2.cvtColor(sample_img,cv2.COLOR_BGR2RGB))

#check if any landmarks are found
if results.pose_landmarks:
    
    #iterate 2 times as we only want to display first 2 landmarks
    for i in range(33):
        
        #display the found normalized landmarks
        print(i)
        print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')
        
        

#z co-ordinate is obtainde by fuguring out relative lenght of a part w.r.t different camera angles






#retrieve height and widhht of the image
image_height, image_width, _ = sample_img.shape

#check if landmarks are found
if results.pose_landmarks:
    
    #iterate two times as we only want to display first 2 landmarks
    for i in range(32):
        
        #display the found landmarks after converting them into their original scale
        print(f'{mp_pose.PoseLandmark(i).name}:')
        print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
        print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_width}')  
        print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
        print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility * image_width}')
        
        
        


#drawing the detected landmarks on the sample image

#create a copy of the sample image to draw landmarks on
img_copy = sample_img.copy()

#check if any landmarks are found
if results.pose_landmarks:
    
    #draw pose landmarks on sample image'
    mp_drawing.draw_landmarks(image = img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
    
    #specify a size of the figure
    fig = plt.figure(figsize = [10,10])
    
    #display the output image with the landmarks drawn, also convert BGR to RGB for display
    plt.title("Output");plt.axis('off');plt.imshow(img_copy[:,:,::-1]);plt.show()
        
        

#plot pose landmarks in 3d
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
#pose_world_landmarks is a list containing 3d pose landmarks
#it sets hip landmark as origin and calculates relative distance of other points from hip
#as this distance increases or decreses depending on how close u are to the camera, it gives us a sens eof depth






#create a pose detection function
def detectPose(image, pose, display = True):
    
    #create a copy of the input image to draw landmarks on
    output_image = image.copy()
    
    #convert the image from BGR to RGB
    imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    #perform the pose detection
    results = pose.process(imageRGB)
    
    #retrieve height and widhht of the image
    height, width, _ = imageRGB.shape 
    
    #initialize a list to store detected landmarks
    landmarks = []
    
    #check if any landmarks are detected
    if results.pose_landmarks:
        
        #draw pose landmarks on output image
        mp_drawing.draw_landmarks(image = output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        
        i=0
        #iterate over the detectecd landmarks
        for landmark in results.pose_landmarks.landmark:
            
            
            #append the landmark into the list
            landmarks.append([landmark.x *width, landmark.y *height, landmark.z *width] )
            #landmarks.append([landmark.x , landmark.y , landmark.z ] )
            i+=1
    
    #check if the original input image and the resultant image are specified to be displayed
    if display:
        
        #display the original input image and the resultant image
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output image");plt.axis('off');
        
        #also plot the pose landmarks in 3d
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    #otherwise
    else:
        
        #return the output image and the found landmarks
        return output_image, landmarks
    
    

from cv2 import dnn_superres
#read a sample image and perform pose detection on it
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()
# Read the desired model
path = r"D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\EDSR_x4.pb"
sr.readModel(path)
# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)


final_list_x = []

for j in range(2,14):

    list_x = []
    
    image_path = r"D:\dev\projects\football AR\testing\results\shots\shot_" + str(j)
    image_list = os.listdir(image_path)
    i=0
    for image in image_list:
    
        img = cv2.imread(os.path.join(image_path,image))

    
        plt.imshow(img)


        # Upscale the image
        image = sr.upsample(img)
        #plt.imshow(image)

        list2 = []
        op_image,list2 = detectPose(image, pose, display = False)
        
        plt.imshow(op_image)
        
        list_x.append(list2)
    
        print(i)
        i+=1











    #making the shooting action dataset
    #includes frame cutouts from actual bird eye view matches(after passing through super resolution)
    #and regular shooting action images as found online
    #primary_list = []
    #this list contains all the shot and non shot objects 
    #and must be converted to a pandas dataframe/csv file later on
    
    #an object consisits of x and y and z coordinates of each landmark(excluding face and fingers)
    #as independant variables
    #dependant variable is 1, inidicating shot and 0, indicating a no shot
    #eg : object - [(11)LEFT_SHOULDER(x,y,z), (12)RIGHT_SHOULDER(x,y,z), (13)LEFT_ELBOW(x,y,z), (14)RIGHT_ELBOW(x,y,z), 
    #(15)LEFT_WRIST(x,y,z),(16)RIGHT_WRIST(x,y,z), (23)LEFT_HIP(x,y,z), (24)RIGHT_HIP(x,y,z), (25)LEFT_KNEE(x,y,z), (26)RIGHT_KNEE(x,y,z), 
    #(27)LEFT_ANKLE(x,y,z), (28)RIGHT_ANKLE(x,y,z), (29)LEFT_HEEL(x,y,z), (30)RIGHT_HEEL(x,y,z), (31)LEFT_FOOT_INDEX(x,y,z), 
    #(32)RIGHT_FOOT_INDEX(x,y,z), Y]     
    
    #total 48 independant variables and 1 dependent variable
    
                                     
    #put this in a for loop containing all the shot and non shot images
    
    #read a sample image and perform pose detection on it
    #image = cv2.imread(r'C:\Users\Asus\Desktop\player.jpg')
    #_,list2 = detectPose(image, pose, display = False)

    list1 = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
    
    
    for j in range(len(list_x)):
        secondary_list = []
        for i in range(len(list_x[j])):
            
            if list1[i] == 1:
                      secondary_list.append(list_x[j][i][0])
                      secondary_list.append(list_x[j][i][1])
                      secondary_list.append(list_x[j][i][2])
    
            else:
                pass
    
        secondary_list.append(1)
        final_list_x.append(secondary_list)


column = ["LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "LEFT_SHOULDER_z", 
          "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y", "RIGHT_SHOULDER_z", 
          "LEFT_ELBOW_x",  "LEFT_ELBOW_y", "LEFT_ELBOW_z",
          "RIGHT_ELBOW_x", "RIGHT_ELBOW_y", "RIGHT_ELBOW_z",
          "LEFT_WRIST_x", "LEFT_WRIST_y", "LEFT_WRIST_z",
          "RIGHT_WRIST_x", "RIGHT_WRIST_y", "RIGHT_WRIST_z",
          "LEFT_HIP_x", "LEFT_HIP_y", "LEFT_HIP_z",
          "RIGHT_HIP_x", "RIGHT_HIP_y", "RIGHT_HIP_z",
          "LEFT_KNEE_x", "LEFT_KNEE_y", "LEFT_KNEE_z",
          "RIGHT_KNEE_x", "RIGHT_KNEE_y", "RIGHT_KNEE_z",
          "LEFT_ANKLE_x", "LEFT_ANKLE_y", "LEFT_ANKLE_z",
          "RIGHT_ANKLE_x", "RIGHT_ANKLE_y", "RIGHT_ANKLE_z",
          "LEFT_HEEL_x", "LEFT_HEEL_y", "LEFT_HEEL_z",
          "RIGHT_HEEL_x", "RIGHT_HEEL_y", "RIGHT_HEEL_z",
          "LEFT_FOOT_INDEX_x", "LEFT_FOOT_INDEX_y", "LEFT_FOOT_INDEX_z",
          "RIGHT_FOOT_INDEX_x", "RIGHT_FOOT_INDEX_y", "RIGHT_FOOT_INDEX_z",
          "Y",     
          ]


dataset = pd.DataFrame(final_list_x, columns= column)
dataset['Y'] = dataset['Y'].replace([False],1)


dataset.to_csv(r"D:\dev\projects\football AR\testing\dataset\dataset1_shot.csv")

dataset_new = pd.read_csv(r"D:\dev\projects\football AR\testing\dataset\dataset1_noshot.csv")
dataset_new['Y'] = dataset_new['Y'].replace(1,0)
dataset_new.to_csv(r"D:\dev\projects\football AR\testing\dataset\dataset1_noshot.csv")     
dataset_total = pd.concat([dataset, dataset_new], ignore_index=True)
dataset_total.to_csv(r"D:\dev\projects\football AR\testing\dataset\dataset1_total_12_instances.csv")

##########################################################################################################
#LSTM model



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
import random

dataset = pd.read_csv(r"D:\dev\projects\football AR\testing\dataset\dataset1_total_12_instances.csv")
dataset = dataset.drop([30,46], axis = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()





#49TH index is Y, which is not included, so 48 columns are included
x_train_shot = dataset.iloc[:120,1:49]
x_train_no_shot = dataset.iloc[180:300,1:49]
x_train = pd.concat([x_train_shot,x_train_no_shot], ignore_index=True)
x_train = x_train.values
#x_train = sc.fit_transform(x_train)

x_test_shot = dataset.iloc[120:180,1:49]
x_test_no_shot = dataset.iloc[300:,1:49]
x_test = pd.concat([x_test_shot,x_test_no_shot], ignore_index=True)
x_test = x_test.values
x_test = sc.transform(x_test)

y_train_shot = dataset.iloc[:120,49:]
y_train_no_shot = dataset.iloc[180:300,49:]
y_train = pd.concat([y_train_shot,y_train_no_shot], ignore_index=True)
y_train = y_train.values

y_test_shot = dataset.iloc[120:180,49:]
y_test_no_shot = dataset.iloc[300:,49:]
y_test = pd.concat([y_test_shot,y_test_no_shot], ignore_index=True)
y_test = y_test.values


#(30,48) will have 30*48 elements while (30,15,48) will have 30*15*48 elements
#break everything into batch
#i.e. when reshaping no. of elements should always be same
#therefore reshape (30,48) into (2,15,48)

#x_train contains 2 sequence instances, each sequence instance is 15units long
x_train=x_train.reshape(16,15,48)


x_train_shuffle_assisstant = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
random.shuffle(x_train_shuffle_assisstant)
x_train_shuffled = []
for i in x_train_shuffle_assisstant:
    x_train_shuffled.append(x_train[i])

x_train_shuffled = np.array(x_train_shuffled)


#x_test contains 1 sequence instance, each sequence instance is 15units long
x_test=x_test.reshape(8,15,48)


x_test_shuffle_assisstant = [0,1,2,3,4,5,6,7]
random.shuffle(x_test_shuffle_assisstant)
x_test_shuffled = []
for i in x_test_shuffle_assisstant:
    x_test_shuffled.append(x_test[i])

x_test_shuffled = np.array(x_test_shuffled)


#reshape y_train to reflect the no. of sequence instances
y_train = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])


y_train_shuffled = []
for i in x_train_shuffle_assisstant:
    y_train_shuffled.append(y_train[i])
y_train_shuffled = np.array(y_train_shuffled)


#same for y_test
y_test = np.array([1,1,1,1,0,0,0,0])


y_test_shuffled = []
for i in x_test_shuffle_assisstant:
    y_test_shuffled.append(y_test[i])
y_test_shuffled = np.array(y_test_shuffled)



x = df_concat = pd.concat([x_train, y_train], axis=1)

#input_shape = (time_steps,no_of_columns_in_X_train)
#input_shape = [16,48]

data_dim = 48
timesteps = 15

model = Sequential()
model.add(LSTM(256, input_shape=(timesteps,data_dim), activation='relu', return_sequences=True,stateful = False))
model.add(Dropout(0.2))

model.add(LSTM(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))
    




#Compiling the network
model.compile( loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'] )


#Fitting the data to the model
model.fit(x_train_shuffled,
         y_train_shuffled,
          epochs=50,
          validation_data=(x_test_shuffled, y_test_shuffled))


  














































###########################################################################################################







#pose detection on real time webcam/ video feed

#setup a pose function for video
#try always tweaking min_detection_confidence
pose_video = mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.5, model_complexity = 2)

#initialize the video capture object to read from the webcam
#video = cv2.VideoCapture(1)

#create a named window for resizing purposes
cv2.namedWindow('Pose Detection',cv2.WINDOW_NORMAL)

#initialize the video capture object to read from a video stored in the disk
video = cv2.VideoCapture(r'C:\Users\Asus\Desktop\football AR\raw football videos\FC Barcelona vs Real Madrid (1-2) J31 2015_2016 - FULL MATCH.mp4')

#set video camera size
video.set(3,1280)
video.set(4,960)

#initialize a variable to store the time of the previous frame
time1 = 0

#iterate until the video is accessed successfully
while video.isOpened():
    
    #read a frame
    ok, frame = video.read()
    
    #check if frame is not read properly
    if not ok:
        
        #break the loop
        break
    
    #flip the frame horizontally for natural (selfie-view) visualization
    frame = cv2.flip(frame,1)
    
    #get the width and height of the frame
    frame_height, frame_width, _ = frame.shape
    
    #resize the frame while keeping the aspect ratio
    frame = cv2.resize(frame, (int(frame_width *(640/frame_height)), 640))
    
    #perform the pose landmark detection
    frame, _ = detectPose(frame, pose_video, display = False)
    
    #set the time for this frame to current time
    time2 = time()
    
    #check if the difference between the previous and this frame time > 0 to avoid division by 0
    if(time2 - time1) > 0:
        
        #calculate the number of frames per second
        frame_per_second = 1.0/(time2-time1)
        
        #write the calculated number of frames per second on the frame
        cv2.putText(frame, 'FPS: {}'.format(int(frame_per_second)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        
    #update previous frame time to this frame time
    #as this frame will become previous frame in next iteration
    time1 = time2
    
    #display the frame
    cv2.imshow('Pose detection', frame)
    
    #wait until a key is pressed
    #retrieve the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF

    #check if 'ESC' is pressed
    if(k == 27):
        
        #break the loop
        break
    
#release the video capture object
video.release()

#close the windows
cv2.destroyAllWindows()
    












#pose detection using openPose

net = cv2.dnn.readNetFromTensorflow(r"C:\Users\Asus\Downloads\graph_opt (1).pb")

inWidth = 384
inHeight = 384
thr = 0.2


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


def pose_estimation(frame):
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
        
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (255, 200, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame
 
        
      
image = cv2.imread(r"D:\dev\projects\football AR\testing\shots\shot8_frames\49.jpg")
image= sr.upsample(image)
image = pose_estimation(image)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    
    





