import os
import matplotlib.pyplot as plt


image_path = r"D:\dev\projects\SPAM research\ad images bulk google image downloads\NON-SPAM"
image_list = os.listdir(image_path)
list_ham = []

for image in image_list:
    
    path = os.path.join(image_path,image)
    list_ham.append(path)
    
    
import random


random.shuffle(list_ham)


import cv2

i = 0
for link in list_ham:
    
    try:

        picture = cv2.imread(link)  
        name = r"D:\dev\projects\SPAM research\ad images bulk google image downloads\shuffled_nonspam" + r"\NON_SPAM_" + str(i) + ".jpg"
        cv2.imwrite(name,picture)
        i+=1
        
    except:
        print(i)
        i+=1


