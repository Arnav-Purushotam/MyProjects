#pre processing the ad images

import os
import cv2


#rename and convert to .jpg

folder_path = r"D:\dev\projects\SPAM research\ad images bulk google image downloads\new"
image_list = os.listdir(folder_path)

i =0
for image in image_list:
    
    #img = cv2.imread(os.path.join(image_path,image))
    image_path = os.path.join(folder_path,image)
    img = cv2.imread(os.path.join(folder_path,image))
    
    try:
        
        new_image_path = "D:\dev\projects\SPAM research\ad images bulk google image downloads\total_ads_folder" + str(i) + ".jpg"
        cv2.imwrite(new_image_path,img)
    except:
        print(i)
    i+=1
    
    
    