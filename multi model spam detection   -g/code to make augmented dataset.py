#creating improved augmented dataset


# from PIL import Image
# import cv2


# #resizing spam images to ham size and adding noise

from wand.image import Image

try:
    from PIL import Image
except ImportError:
    import Image

import os
# Read image using Image() function

from PIL import ImageFilter



import numpy as np
import os
import cv2
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss * var*4
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p*8)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p)*5)
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy
  






   

k = 0
pathb = r"D:\SPAM DATASET\augmented_dataset\spam_250"
folder_listb = os.listdir(pathb)

patha = r"D:\SPAM DATASET\augmented_dataset\ham_32"
folder_lista = os.listdir(patha)
j=0
for itema in folder_lista:
    filename1 = os.path.join(patha,itema)
    f1 = patha + '\\' + itema
    name1 = r"D:\SPAM DATASET\new_dataset\aug_ham_on\aug_ham" + str(j) + ".jpg"
    img = cv2.imread(filename1, cv2.IMREAD_COLOR)  
    noise_img =noisy("s&p",img)
    noise_img = cv2.resize(noise_img, (256,256))
    cv2.imwrite(name1, noise_img)
    j+=1


    i = 0

    for itemb in folder_listb:
        filename2 =os.path.join(pathb,itemb)
        f2 = pathb + '\\' + itemb
        with Image.open(f2) as img:
            img = img.resize((256,256))
            name2 = r"D:\SPAM DATASET\new_dataset\aug_spam_on\aug_spam" + str(i) + ".jpg"
            img.save(name2)
            i+=1


        background = Image.open(name1)
        overlay = Image.open(name2)
    
        background = background.convert("RGBA")
        overlay = overlay.convert("RGBA")

        new_img = Image.blend(background, overlay, 0.57)
        name = r"D:\SPAM DATASET\augmented_dataset_3\aug_spam_" + str(k) + ".png"
        new_img = new_img.save(name)
        print("saved " + str(k))
    
        k+=1












    


