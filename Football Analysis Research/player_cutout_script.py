import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib
#from cv2 import dnn_superres


#superresolution function












#tracking the ball




















#41,               48,               52


#tracking players
#make sure to crop the video before tracking


TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
         'kcf' : cv2.legacy.TrackerKCF_create,
         'boosting' : cv2.legacy.TrackerBoosting_create,
         'mil': cv2.legacy.TrackerMIL_create,
         'tld': cv2.legacy.TrackerTLD_create,
         'medianflow': cv2.legacy.TrackerMedianFlow_create,
         'mosse':cv2.legacy.TrackerMOSSE_create,
         
}

trackers = cv2.legacy.MultiTracker_create()


v = cv2.VideoCapture(r"D:\dev\projects\football AR\testing\shots\shot46\shot52_sjELtHsL.mp4")


ret, frame = v.read()


k = 1
for i in range(k):
    
    cv2.imshow('Frame',frame)
    bbi = cv2.selectROI('Frame',frame)
    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i,frame,bbi)
    
    
    
frameNumber = 2
#baseDir = r'C:\Users\Asus\Desktop\football AR\code\tracking_results'

#list1 to store frame co-ordinates and frame number
list1 = []
#list 2 to store the cropped frames
list2 = []









from cv2 import dnn_superres
#read a sample image and perform pose detection on it
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()



# Read the desired model
path = r"D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\EDSR_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)




while True:
    ret, frame = v.read()
    if not ret:
        break
    (success,boxes) = trackers.update(frame)
    list1.append([frameNumber,boxes])
    #cropped = img[start_row:end_row, start_col:end_col]
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        #image = cv2.circle(frame, (x,y), radius=0, color=(0, 0, 255), thickness=5)
        cropped = frame[y:y+h,x:x+w]
        #if you want to see the cropped image
        #plt.imshow(cropped)
        list2.append(cropped)


    frameNumber+=1
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    
    
v.release()
cv2.destroyAllWindows()





errors = []
folder = r"D:\dev\projects\football AR\testing\shots\shot46"
for i in range(len(list2)):
    
    try:
        # Upscale the image
        image = list2[i]
    
    
        name = str(i) + ".jpg"
        abs_file_path = os.path.join(folder, name)
    
        cv2.imwrite(abs_file_path, image)
    
        #k = np.array(list2[i])


        #im = Image.fromarray(k)
    
    
    
        #im.save(abs_file_path)
    except:
        errors.append(i)
        








#determining player id and team