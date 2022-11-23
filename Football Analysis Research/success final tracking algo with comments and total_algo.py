import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from numpy import argmax
import math


#imports for the rsc-net model
import argparse
import imageio
import scipy.misc
from models import hmr, SMPL
#import hmr, SMPL
import config, constants
import torch
from torchvision.transforms import Normalize
import numpy as np
#from utils.renderer import Renderer
import os
import pyvista as pv
import cv2



#loading the trained model (not transfer learning)
model = load_model(r'D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\arabic video files\files-20220720T172825Z-001\files\model.h5')

#loading the video
cap = cv2.VideoCapture(r'D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\arabic video files\data-20220720T172829Z-001\data\video.avi')

#loading the pixelatedd(zoomed in) ball image 
temp=cv2.imread(r'D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\arabic video files\data-20220720T172829Z-001\data\temp.jpg',0)
#loading the pitch image for 2d projection
ground=cv2.imread(r'D:\dev\projects\football AR\code\multiple-object-tracking-in-videos-using-opencv-in-python\arabic video files\data-20220720T172829Z-001\data\dst.jpg')


#The height of the image is stored at the index 0.
#The width of the image is stored at index 1.
#The number of channels in the image is stored at index 2
#below line iterates through the shape list backwards hence ignores number of channels
wt, ht = temp.shape[::-1]


#checking if video opening is successfull or not
if (cap.isOpened()== False): 
    print("Error opening video stream or file")


# Load Yolo for players
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

colors = np.random.uniform(0, 255, size=(len(classes), 3))     
    






#load custom yolo for ball

# Load Yolo
net_ball = cv2.dnn.readNet(r"D:\dev\projects\football AR\code\football ball tracking dataset\train_yolo_to_detect_custom_object\yolo_custom_detection\yolov3_training_last.weights", r"D:\dev\projects\football AR\code\football ball tracking dataset\train_yolo_to_detect_custom_object\yolo_custom_detection\yolov3_testing.cfg")

# Name custom object
classes_ball = ["b"]




layer_names_ball = net_ball.getLayerNames()
output_layers_ball = [layer_names_ball[i-1] for i in net_ball.getUnconnectedOutLayers()]
colors_ball = np.random.uniform(0, 255, size=(len(classes_ball), 3))









def plane(players,ball):
    coptemp=ground.copy()
    matrix=np.array([[ 2.56945407e-01,  5.90910632e-01,  1.94094537e+02],
                     [-1.33508274e-02,  1.37658562e+00, -8.34967286e+01],
                     [-3.41878940e-05,  1.31509536e-03,  1.00000000e+00]])
    
    for p in players:
        x=p[0]+int(p[2]/2)
        y=p[1]+p[3]
        pts3 = np.float32([[x,y]])
        pts3o=cv2.perspectiveTransform(pts3[None, :, :],matrix)
        x1=int(pts3o[0][0][0])
        y1=int(pts3o[0][0][1])
        pp=(x1,y1)
        if(p[4]==0):
            cv2.circle(coptemp,pp, 15, (255,0,0),-1)
        elif p[4]==1:
            cv2.circle(coptemp,pp, 15, (255,255,255),-1)
        elif p[4]==2:
            #print hakm
            #cv2.circle(coptemp,pp, 15, (0,0,255),-1)
            pass
    if len(ball) !=0:
        
        xb=ball[0]+int(ball[2]/2)
        yb=ball[1]+int(ball[3]/2)
        pts3ball = np.float32([[xb,yb]])
        pts3b=cv2.perspectiveTransform(pts3ball[None, :, :],matrix)
        x2=int(pts3b[0][0][0])
        y2=int(pts3b[0][0][1])
        pb=(x2,y2)
        cv2.circle(coptemp,pb, 15, (0,0,0),-1)
    return coptemp







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






#the below list holds all the frames in their own format
all_frames_list = []


opr=0
frame_list = []
while(cap.isOpened()):
    ret, frame = cap.read()
    
    #players list contains all the player co-ordinates + team in that frame
    players=[]
    #ball list contains the ball co-ordinates in that frame
    ball=[]
    if opr<310:
        opr=opr+1
        continue
    
    
    if ret == True :
        
        #making a copy of the frame
        copy=frame.copy()
        
        all_frames_list.append(copy)
        
        #converting to greyscale, the original frame
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
            #getting that particular cutout
            roi = frame[y:y+h,x:x+w]
            
            #some frames are bad so resize function throw an error
            try:
                #resizing the frame so as to feed the model
                roi=cv2.resize(roi, (96,96))
            except:
                continue
            #further resizing so as to form input to the model
            #this model predicts the team using shirt color
            #can be put outside alongwith shirt number identifier to increase initial tracking performance
            ym=model.predict(np.reshape(roi,(1,96,96,3)))
            #obtain the detected class, ym is actually probabilities
            #argmax obtains a final class from all the (three) probabilities
            ym=argmax(ym)
            
            #a list having the players co-ordinates andd team members
            players.append([x,y,w,h,ym])
            
            #opencv uses BGR format
            #drawing the player rectangles on the image copy which we have made
            #(detection of players is made using the grayscaled image which is converted to blob and fed into the yolo network)
            if ym==0:
                #red rectangle, for team A players
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,0,255), 2)
            elif ym==1:
                #green rectangle, for team B players
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,255,0), 2)
            elif ym==2:
                #blue rectangle, for referee
                cv2.rectangle(copy, (x, y), (x + w, y + h), (255,0,0), 2)
            
        
        #tracking the ball in the code below
        #template matching method present in opencv to detect parts of an image
        #It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image
        #temp is the ball image which we have loaded in
        #cv2.TM_SQDIFF_NORMED is a sort of detector
        #res = cv2.matchTemplate(gray,temp,cv2.TM_SQDIFF_NORMED)
        #cv.TM_SQDIFF_NORMED as comparison method, minimum value gives the best match.
        #template image is of size (wxh) (the ball)
        #Once you got the result, you can use cv.minMaxLoc() function to find where is the maximum/minimum value. Take it as the top-left corner of rectangle and take (w,h) as width and height of the rectangle
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #if min_val < 0.05:
            #top_left = min_loc
            #top_left is a list containing x at 0th position and y at 1st positon
            #getting the bottom right co-ordinate by adding widht to x and height to y
            #height increases downwards so adding the height not subtracting
            #bottom_right = (top_left[0] + wt, top_left[1] + ht)
            #appending to ball list (x,y,w,h)
            #ball.append(top_left[0])
            #ball.append(top_left[1])
            #ball.append(wt)
            #ball.append(ht)
            #drawing the ball rectangle on the image copy
            #cv2.rectangle(copy,top_left, bottom_right, (0,255,100), 2)
            
        #the below line calls a function which constructs the 2d template for that frame
        #can be commented out to increase initial tracking performance
        #p=plane(players, ball)
            
        #out this is the final output image containing rectangles around ball and players
        #out.write(copy)
        
        #out2 this is the final image of 2d projection
        #out2.write(p)
        
        
        
        #custom yolo model to detect thee ball
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net_ball.setInput(blob)
        outs = net_ball.forward(output_layers)
        
        
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
      
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                #ball list contains co-ordinates of the ball
                ball.append([x, y, w, h])
                label = str(classes_ball[class_ids[i]])
                color = colors_ball[class_ids[i]]
                cv2.rectangle(copy, (x, y), (x + w, y + h), color, 2)
                cv2.circle(copy, (x,y), radius=0, color=(0, 255, 255), thickness=-1)
                #cv2.putText(copy, label, (x, y + 30), font, 3, color, 2)      
                cv2.circle(copy, (int(x+w/2),int(y+h/2)), radius=0, color=(0, 255, 255), thickness=-1)
                
    
    

        
                    
                    


        
        
            
            
        
        
        
        
        
        #showing the respective images
        cv2.imshow('img',copy)
        #cv2.imshow('plane',p)
        

        #frame_list contains list instances corresponding to each frame, this instance contains 
        #players list (a list of all players in that frame) and ball list (containing ball co-ordinates in that frame)
        frame_list.append([players,ball])

        
    # this will run the video without stop and maybe the cv2 window will stop between every frame
    # depending on your pc power ( i recommend to use (opencv with gpu) and colab to run script quickly ) 
    # if you want script stop between every frame and manually you allow the script to continue change it t ocv2.waitKey(0)
    if cv2.waitKey(1)==27:
        
        break
    
    
    
    
    
#analysis


#remove all dashes in variable names in python as they are not allowed
#do everything in relative distance i.e. pixel values initially
#then compare ball pixel size and actual ball size to get the real distances


















def findClosestPlayerToBall(players_list,ball_center_position_current_frame):
    
    #here measure the distance to the ball from the center of thr feet (or) lower most co-ordinate of player
    #(higher in this case because co-ordinate system here increases downwards)
    #measuring from the bottom left most is not correct as that is just the positon of corner of rectangle
    #but not the actual position of players legs
    
    distances_player_to_ball_list = []
    ball_point = ball_center_position_current_frame
    
    

    for i in range(len(players_list)):
        #player_point = [x+w/2, y+h], considering base of the player here
        player_point = [players_list[i][0] + players_list[i][2]/2  ,  players_list[i][1] + players_list[i][3]]
    
        #now finding distance from player to ball using sqrt((x1-x2)^2 + (y1-y2)^2)
        distance_player_to_ball = math.sqrt(  (((player_point[0] - ball_point[0])^2)  +  ((player_point[1] - ball_point[1])^2))  )
    
        distances_player_to_ball_list.append(distances_player_to_ball_list)
    
    
    
    min_distance = min(distances_player_to_ball_list)
    
    #finding co-ordinates of the closest player to the ball
    index_of_min_distance_in_distances_list = distances_player_to_ball_list.index(min_distance)
    closest_player_to_ball_co-ordinates = players_list[index_of_min_distance_in_distances_list]
    
    return min_distance,closest_player_to_ball_co-ordinates
    
    


#lists are passed by reference in python
def obtainCutout(closest_player_co-ordinates,frame_to_be_sent):
    
    
    
    (x,y,w,h) = closest_player_co-ordinates
    cropped = frame_to_be_sent[y:y+h,x:x+w]
    
    return cropped
    
    





#first test if its possible to get those images without execution in cmd
#and by just written code in python

#to connect the lstm model to the total algo, both have to be in the same environment
#because the code below will have to call functions that call the model, so
#as you see its a single execution thread, i.e. only 1 environment is present
#first clone tf2.5 environment and name it football
#then install in football all the packages installed in RSC-net
#then execute the total algo in football environment

#create a function to first get the segmentation of players by using RSC-net
#then get the keypoints from those images
#shape the data in the format that is required for the lstm model then feed it in


#save the lstm model
#load the model
#shape data into correct input format for model and call t






def size_to_scale(size):
    if size >= 224:
        scale = 0
    elif 128 <= size < 224:
        scale = 1
    elif 64 <= size < 128:
        scale = 2
    elif 40 <= size < 64:
        scale = 3
    else:
        scale = 4
    return scale










def RSCNETFunction(video):
    
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    hmr_model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(checkpoint_path)
    hmr_model.load_state_dict(checkpoint, strict=False)
    hmr_model.eval()
    hmr_model.to(device)

    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    #img_renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)
    
    #fi = []
    joints_all_frames = []
    
    for image in image_list:
        joints = []
    
        #img_path = os.path.join(folder_path,image)
    
        #img = imageio.imread(img_path)
        im_size = img.shape[0]
        im_scale = size_to_scale(im_size)
        img_up = scipy.misc.imresize(img, [224, 224])
        img_up = np.transpose(img_up.astype('float32'), (2, 0, 1)) / 255.0
        img_up = normalize_img(torch.from_numpy(img_up).float())
        images = img_up[None].to(device)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera, _ = hmr_model(images, scale=im_scale)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            #print(pred_output)
            #pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints
            #fi.append(pred_vertices.cpu().numpy())
            joints.append(pred_joints.cpu().numpy())
            joints = joints[0][0]
            final_joints = []
            for i in range(25):
                
                #excluding right eye, left eye, right ear, left ear
                if i not in [15,16,17,18]:
                    final_joints.append(joints[i])
                    
            joints_all_frames.append(final_joints)
            
    return joints_all_frames
    
    






def data_reshaper(all_frames_joints):
    
    







def LSTMpredictor(input_data):
    






def real_shot_checker(video):
    
    #get the x,y,z of all co-ordinates
    unshaped_keypoints_all_frames = RSCNETFunction(video)
    
    #now shape the co-ordinates to fit as input to lstm network
    LSTMInputData = data_reshaper(unshaped_keypoints_all_frames)
    
    #then feed it to the lstm network
    boolean = LSTMpredictor(LSTMInputData)
    
    return boolean














#video here is just a list of frames
def check_shooting_action(video):
    
    #video = [video_list,last_frame_number_of_video]
    #video_list is a list of elements with each element as:
    #[cutout_img,frame_number,closest_player_co-ordinates,ball_center_position_current_frame, "possession"]
    
    video_only = []
    video_list = video[0]
    for i in range(len(video_list)):
        
        #video_only has only frames
        video_only.append(video_list[i][0])
        
        
    
    
    
    if len(video_only) >= 15
        
        if len(video_only) == 15:
            #pass all the frames to the model
            #video[1] is the frame number as video = [video_list,last_frame_number_of_video]
            start_frame_of_shot = video[1]- 15
            
            
            
        else:
            #slicing backwards to get the last 15 elements in the list
            video_to_send = video_only[-15:]
            #if last frame is 70 then 70-15 = 55, so frame with index 55 is start 
            #frame of the shot
            start_frame_of_shot = video[1] - 15
            #pass the frames to the model
            
            boolean = real_shot_checker(video_to_send)
            
        return start_frame_of_shot,boolean
            
        
    else:
        return "no shot", False
    










def check_for_pass(video,pass_list,intercept_list):
    
    #get the team of the player in that video
    #get the next frame after the last frame of the video
    #by indexing through action list
    #keep appending those frames to a list until the action involved is possession
    #now you have a list of all frames right after a loss of possession(which is not a shot)
    #check the next frame which is a possession frame
    #check the team of the closest player in that frame
    #if final team and initial team are the same then it is a pass
    #else it is an interception
    #append pass to pass_list , its format is [pass_video,start co-ordinate of pass,end co-ordinate of pass, team involved, passer involved,reciever involved, distance of pass, velocity of pass in each frame]
    #append interception to interception_list, its format is [interception_video,start co-ordinate of interception,end co-ordinate of interception, team and player which made the bad pass, team and player which made the interception]
    
    
    initial_team = video[0][0][2][4]
    starter_co-ordinates = video[0][0][2]
    last_frame_of_video = video[1]
    pass_or_intercept_list = []
    i = last_frame_of_video + 1
    #appending all the out of possession frames to a list
    while len(action_list[i]) == 4:
       pass_or_intercept_list.append(action_list[i])
       i+=1
       
    final_team = action_list[i+1][2][4]
    
    if final_team == initial_team:
        #it is a pass
        start_co-ordinate_of_pass = pass_or_intercept_list[0][0]
        end_co-ordinate_of_pass = pass_or_intercept_list[-1][0]
        passer_involved = action_list[last_frame_of_video][2]
        reciever_involved = action_list[i+1][2]
        team_involved = initial_team
        distance_of_pass = math.sqrt(  (((start_co-ordinate_of_pass[0] - end_co-ordinate_of_pass[0])^2)  +  ((start_co-ordinate_of_pass[1] - end_co-ordinate_of_pass[1])^2))  )
        
        ball_velocities = []
        for i in range(len(pass_or_intercept_list)):
            ball_velocities.append(pass_or_intercept_list[i][2])
            
        
        
        pass_list = [pass_or_intercept_list, start_co-ordinate_of_pass, end_co-ordinate_of_pass, passer_involved, reciever_involved, team_involved, distance_of_pass,  ball_velocities]
        return True, intercept_list
        
    else:
        
        start_co-ordinate_of_intercept = pass_or_intercept_list[0][0]
        end_co-ordinate_of_intercept = pass_or_intercept_list[-1][0]
        passer_involved = action_list[last_frame_of_video][2]
        reciever_involved = action_list[i+1][2]
        team_involved_in_bad_pass = initial_team
        team_that_intercepted = final_team
        distance_of_intercept = math.sqrt(  (((start_co-ordinate_of_intercept[0] - end_co-ordinate_of_intercept[0])^2)  +  ((start_co-ordinate_of_intercept[1] - end_co-ordinate_of_intercept[1])^2))  )
        
        ball_velocities = []
        for i in range(len(pass_or_intercept_list)):
            ball_velocities.append(pass_or_intercept_list[i][2])
         
        
        
        
        
        
        intercept_list = [pass_or_intercept_list, start_co-ordinate_of_intercept, end_co-ordinate_of_intercept, passer_involved, reciever_involved, team_involved_in_bad_pass, team_that_intercepted, distance_of_intercept, ball_velocities]
        return False,intercept_list
        
        
    














#put these 5 outside the forloop
action_list_total = []
action_list_possession = []
action_list_out_of_possession = []
frames_per_sec = 50
time_btw_two_frames = 0.02





#primary_list.append(ball_co-ordinates, (those detected in camera only,co-ordinates)[player1,...playern],frame number)
primary_list = frame_list


#starting from 1 as ball_list_previous_frame is accessed in an iteration
for frame_number in range(1,len(frame_list)):
    players_list = primary_list[frame_number][0]
    ball_list_current_frame = primary_list[frame_number][1]
    ball_list_previous_frame = primary_list[frame_number - 1][1]
    ball_center_position_current_frame = [int(ball_list_current_frame[0] + ball_list_current_frame[2]/2),int(ball_list_current_frame[1] + ball_list_current_frame[3]/2)]
    ball_center_position_previous_frame = [int(ball_list_previous_frame[0] + ball_list_previous_frame[2]/2),int(ball_list_previous_frame[1] + ball_list_previous_frame[3]/2)]






    #instantaneous_ball_velocity_of_frame_0 = 0
    #instantaneous_ball_velocity = (ball_co-ordinates in frame(n) - ball_co-ordinates in frame(n-1)) / (1/frames_per_second_of_original_video(it is a constant))
    #(x,y) is top left corner, the co-ordinate system increases downwards and increases towards right, so centre is (x+w/2,y+h/2)
    #instantaneous_ball_velocity = [x_speed,y_speed,combined_speed(sqrt((x_speed)^2 + (y_speed)^2))]
    x_speed = abs(ball_center_position_current_frame[0] - ball_center_position_previous_frame[0])/time_btw_two_frames
    y_speed = abs(ball_center_position_current_frame[1] - ball_center_position_previous_frame[1])/time_btw_two_frames
    combined_speed = math.sqrt(((x_speed^2)  +  (y_speed^2)))
    instantaneous_ball_velocity = [x_speed,y_speed,combined_speed]









    #finds the closest player to the ball in a particular frame
    #given a list of co-ordinates of all players present in the frame and ball center co-ordinates
    #in that frame as the input
    closest_player_co-ordinates, distance_to_ball = findClosestPlayerToBall(players_list,ball_center_position_current_frame)






    #(both the below conditions should pass, that is ball velocity is very low and in the circle of the player, only then it is considered as possession)
    # k can be obtained by observation
    #player_height is height of the rectangular box
    #if distance_to_ball <= 1/2(player_height)   and   instantaneous_ball_velocity <= k   then:
        #cutout_img = obtainCutout(closest_player_co-ordinates)
        #action_list_possession.append(cutout_img,frame_number,player_co-ordinate,ball_co-ordinate, "possession")
    #otherwise:
        #action_list_out_of_possession.append(ball_co-ordinates,frame_number,"out of possession")

    player_height = closest_player_co-ordinates[3]
    player_team = closest_player_co-ordinates[4]
    # set k1 factor as desired
    k1 = 1
    player_circle_radius = k1*player_height
    #instantaneous_ball_velocity[2] = combined_speed
    if distance_to_ball <= player_circle_radius    and   instantaneous_ball_velocity[2] <= k:
    
        frame_to_be_sent = all_frames_list[frame_number]
        cutout_img = obtainCutout(closest_player_co-ordinates,frame_to_be_sent)
        action_list_possession.append([cutout_img,frame_number,closest_player_co-ordinates,ball_center_position_current_frame, "possession"])
        action_list.append([cutout_img,frame_number,closest_player_co-ordinates,ball_center_position_current_frame,instantaneous_ball_velocity, "possession"])
    
    else:
    
        action_list_out_of_possession.append([ball_center_position_current_frame,frame_number,"out of possession"])
        #ball_center_position_current_frame is a list of the format [x,y]
        action_list.append([ball_center_position_current_frame,frame_number,instantaneous_ball_velocity,"out of possession"])
    
    
    
   
   
   
   
   

#after doing the above for the entire video
#action_list.append(every entry in action_list_possession and every entry in action_list_out_of_possession)
#for all elements in action_list:
    #sort entries according to frame number

# acti = []

# for element in action_list_possession:
#     acti.append(element)

# for element in action_list_out_of_possession:
#     acti.append(element)
    
    
# for i in range(len(acti)):
#     action_list.append(0)
    

# for element in acti:
#     #index 1 of element is frame number
#     index = element[1]
#     action_list[index] = element
    

# #action_list now has frames






    
    

    
#for images in action_list_possession, if frame_number not continuous:
#append to a list those frames which have continous frames
#convert above list to a video == possession_action1.video
#video_action_possession_list = []
#video_action_possession_list.append([possession_action1.video, end_of_video_frame_number])
#do the above for all possession frames


video_action_possession_list = []
possession_video = []
#0th is appended before for loop as at every iteration i+1 is appended and not i
#so 0th frame will be lost if not accounted for
possession_video.append(action_list_possession[0])
for i in range(len(action_list_out_of_possession)-1):
    
    if action_list_possession[i+1][1] = action_list_possession[i][1] + 1:
        possession_video.append(action_list_possession[i+1])
    else:
        #the last frame here in the video is i
        video_action_possession_list.append(possession_video,action_list_possession[i][1])
        possession_video = []
        #this is done as at every instant the next frame is appended to a list
        #so when the else clause is executed, the next frame(which didnt qualify)
        #is never appended to any list and is lost
        possession_video.append(action_list_possession[i+1])
    
    
    
    
    
    






#for images in action_list_out_of_possession, if frames in continous fashion
#OutOfPossessionCo-ordinates_secondary = []
#OutOfPossessionCo-ordinates_primary = []
#append the [continous group of frames, last_frame_number] to OutOfPossessionCo-ordinates_secondary list
#OutOfPossessionCo-ordinates_primary.append(OutOfPossessionCo-ordinates_secondary)

video_action_out_of_possession_list = []
out_of_possession_video = []
#0th is appended before for loop as at every iteration i+1 is appended and not i
#so 0th frame will be lost if not accounted for
out_of_possession_video.append(action_list_out_of_possession[0])
for i in range(len(action_list_out_of_possession)-1):
    
    if action_list_out_of_possession[i+1][1] = action_list_out_of_possession[i][1] + 1:
        out_of_possession_video.append(action_list_out_of_possession[i+1])
    else:
        #the last frame here in the video is i
        video_action_out_of_possession_list.append(out_of_possession_video,action_list_out_of_possession[i][1])
        out_of_possession_video = []
        #this is done as at every instant the next frame is appended to a list
        #so when the else clause is executed, the next frame(which didnt qualify)
        #is never appended to any list and is lost
        out_of_possession_video.append(action_list_out_of_possession[i+1])
    
    
 








#for element in video_action_possession_list
#pass only the last 15 frames of a video to the model
#result = pose_recognition_model.predict(element[0]), 0 has video, 1 has last frame number
#if result == no shot:
    #last_frame = element[1]
    #get OutOfPossessionCo-ordinates_secondary list in OutOfPossessionCo-ordinates_primary list whose last_frame_number = last_frame + 1
    #trajectory,distance = get_trajectory_&_distance(OutOfPossessionCo-ordinates_secondary list)
    

shots = []
pass_list = []
intercept_list = []
for video in video_action_possession_list:
    
    
    
    start_frame_of_shot,shot_action = check_shooting_action(video)
    
    if shot_action == False:
        
        #check for a pass
        v1,v2 = check_for_pass(video,pass_list,intercept_list)
        
        if v1:
            pass_list.append(v2)
        else:
            intercept_list.append(v2)
        
        
    
        
    
    else:
        end_frame_of_shot = video[1]
        
        #2th index is the co-ordinates of the player closest to the ball which has x,y,w,h,team
        player_involved = video[0][0][2]
        
        
        shots.append([video[0],start_frame_of_shot,end_frame_of_shot,player_involved])
        
        
 
        
 
    
 
    
# shots_list_final = []  
# for i in range(len(shots)):
    
#     #add here more info to each shot such as the length, final and start position, velocity of shot in each frame,etc
    
    













    