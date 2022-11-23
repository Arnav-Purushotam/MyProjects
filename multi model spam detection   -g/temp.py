from imageio import imread
import matplotlib.pyplot as plt
import time
import numpy as np
import hashlib


#hashing creates a fingerprint of our image

from hashlib import md5

def file_hash(filepath):
    with open(filepath, 'rb') as f:  #reading the file as f
        return md5(f.read()).hexdigest()  #.hexdigest method forms the fingerprint



import os
os.getcwd()
os.chdir(r'D:\dev\projects\SPAM research\ad images bulk google image downloads\ads_-_Google_Search')  #this must be in single quotes
os.getcwd()
files_list = os.listdir()
print(len(files_list))


duplicates = []
hash_keys = dict()
for index, filename in  enumerate(os.listdir('.')):  #listdir('.') = current directory
#enumerate just counts the iterations and stores it in index
    if os.path.isfile(filename): #to check if our path is indeed a file and not a folder or something else
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys: 
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash])) 
            #here index is the index of the duplicate in our file and 2nd arg is index of the original img in dict
        
        
print(duplicates)        
        

#this is to visualise the duplicates
#left are the originals and right are the duplicates
for file_indexes in duplicates[:30]:
    try:
    
        plt.subplot(121),plt.imshow(imread(files_list[file_indexes[1]]))
        plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])

        plt.subplot(122),plt.imshow(imread(files_list[file_indexes[0]]))
        plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
        plt.show()
    
    except OSError as e:
        continue
        
     
        
for index in duplicates:
    os.remove(files_list[index[0]])
        

       
        
        
        
