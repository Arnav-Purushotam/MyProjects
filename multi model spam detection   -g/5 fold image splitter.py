import cv2
import os

#loading all non spam image links into a list
nospam_image_path = r"D:\dev\projects\SPAM research\ad images bulk google image downloads\shuffled_nonspam"
nospam_image_list = os.listdir(nospam_image_path)
nospamlist = []

for image in nospam_image_list:
    
    img_path = os.path.join(nospam_image_path,image)
    nospamlist.append(img_path)
    
    
#loading all spam image links into a list
spam_image_path = r"D:\dev\projects\SPAM research\ad images bulk google image downloads\shuffled_spam"
spam_image_list = os.listdir(spam_image_path)
spamlist = []

for image in spam_image_list:
    
    img_path = os.path.join(spam_image_path,image)
    spamlist.append(img_path)
    
    



#putting no spam into corresponding ham folders


k = 0
for i in range(1,6):
    
    path_fold = os.path.join(r"D:\dev\projects\SPAM research\DATASETS\5fold helper", str(i)) 
    for j in range(0,1588):
        
        try:
        
            img = cv2.imread(nospamlist[k])
            k+=1
            new_img_path = path_fold + r"\ham" + r"\NOSPAM_" + str(k) + ".jpg"
            cv2.imwrite(new_img_path,img)
        except:
            print(k)
            k+=1
        
        
        
        
        
 
#putting spam into corresponding spam folders       
        
k = 0
for i in range(1,6):
    
    path_fold = os.path.join(r"D:\dev\projects\SPAM research\DATASETS\5fold helper", str(i)) 
    for j in range(0,2119):
        
        try:
        
            img = cv2.imread(spamlist[k])
            k+=1
            new_img_path = path_fold + r"\spam" + r"\SPAM_" + str(k) + ".jpg"
            cv2.imwrite(new_img_path,img)
        except:
            print(k)
            k+=1
   
    
   


from pathlib import Path
#putting data into each of the 5 folds
#manually copy the testing data

part_list = [1,2,3,4,5]
for u in range(1,6):
    
    fold_path = r"D:\dev\projects\SPAM research\DATASETS\5 fold cross validation arranged data" + r"\fold" + str(u) + r"\train"     
    new_part_list = []
    for num in part_list:
        if num != u:
            new_part_list.append(num)
            
    ham_folders_links = []
    spam_folders_links = []    
    ham_img_links = []
    spam_img_links = []
    
    for u1 in new_part_list:
        
        link = os.path.join("D:\dev\projects\SPAM research\DATASETS\5fold helper",str(u1)) 
        link_ham = link + r"\ham"
        ham_folders_links.append(link_ham)
        
        
        link_spam = link + r"\spam"
        spam_folders_links.append(link_spam)
        
        
        
        
    for u2 in range(len(ham_folders_links)):
        
        
        ham_folder = Path(ham_folders_links[u2]).resolve()
        instance_ham_folder_list = os.listdir(ham_folder)
        for image in instance_ham_folder_list:
            img_path = os.path.join(ham_folder,image)
            ham_img_links.append(img_path)
            
    r1 = 0
    for individual_image_link in ham_img_links:
        img = cv2.imread(individual_image_link)
        img_name = "HAM_" + str(r1)
        final_image_path = os.path.join(fold_path,img_name)
        r1+=1
        cv2.imwrite(img,final_image_path)
        
        
    
    
    for u2s in range(len(spam_folders_links)):
        
        spam_folder = Path(spam_folders_links[u2s]).resolve()
        instance_spam_folder_list = os.listdir(spam_folder)
        for image in instance_spam_folder_list:
            img_path = os.path.join(spam_folder,image)
            spam_img_links.append(img_path)
            
    r1s = 0
    for individual_image_link in spam_img_links:
        img = cv2.imread(individual_image_link)
        img_name = "SPAM_" + str(r1s)
        final_image_path = os.path.join(fold_path,img_name)
        r1s+=1
        cv2.imwrite(img,final_image_path)
        
        
        
        
        
        




