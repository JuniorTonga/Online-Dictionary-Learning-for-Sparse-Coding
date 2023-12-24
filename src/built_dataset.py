#https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download&select=seg_test

import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm 
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
from .utils import insert_text_on_image



def get_images(directory,images_classes=['glacier','sea','buildings','forest','street','mountain']):
    images=[]
    labels=[]
    for label in tqdm(os.listdir(directory)):
        if label in images_classes:
            for image_file in os.listdir(directory+label):
                image=cv2.imread(directory+label+r'/'+image_file)
                image=cv2.resize(image,(150,150)) 
                images.append(image)
                labels.append(label)

    shuffle(images,labels)   
    images=np.array(images, dtype=np.float32) / 255.0
    labels=np.array(labels)
    return (images,labels)

def plot_image_and_image_with_text(image, label,text=None):
    f,ax = plt.subplots(5,5) 
    f.subplots_adjust(0,0,3,3)
    for i in range(0,5,1):
        for j in range(0,5,1):
            rnd_number = randint(0,len(image))
            if  text is not None:
                img=insert_text_on_image(image[rnd_number],text)
                img=np.clip(img, 0, 1)
            else :
                img= image[rnd_number]
            ax[i,j].imshow(img)
            ax[i,j].set_title(label[rnd_number])
            ax[i,j].axis('off')









