from PIL import Image
import sys
import os
import cv2
from PIL import UnidentifiedImageError



image_path = r"D:\dev\projects\SPAM research\ad images bulk google image downloads\ads_-_Google_Search"
image_list = os.listdir(image_path)

i = 0

for image in image_list:
    
    try:
        img = cv2.imread(os.path.join(image_path,image))
        name = "image_spam" + str(i) + "kl" + ".jpg"
        Image.open(os.path.join(image_path,image)).convert('RGB').save(name)
        i += 1
    except:
        pass # end of sequence
        

