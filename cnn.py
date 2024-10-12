import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os

#affichage de toutes les fichiers dans la dataset et le nombre des fichier (5856 file)

nombre_de_fichier=0
for dirname, _, filenames in os.walk('C://Users//Mohamed//Desktop//projet image processing//image processing dataset'):
    for filename in filenames :
        print(os.path.join(dirname, filename))
        nombre_de_fichier +=1


print(nombre_de_fichier)




labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)



train = get_training_data('C://Users//Mohamed//Desktop//projet image processing//image processing dataset//train')
test = get_training_data('C://Users//Mohamed//Desktop//projet image processing//image processing dataset//test')
val = get_training_data('C://Users//Mohamed//Desktop//projet image processing//image processing dataset//val')



l = []
for i in train:
    if(i[1] == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)  