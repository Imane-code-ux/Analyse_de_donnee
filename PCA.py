#importing the dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#Import required modules
from sklearn.decomposition import PCA
im=Image.open("lena.png")
T=np.array(im)
blue,green,red = cv2.split(T) 
#initialize PCA with first 20 principal components
pca = PCA(100)
 
#Applying to red channel and then applying inverse transform to transformed array.
red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)
 
#Applying to Green channel and then applying inverse transform to transformed array.
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)
 
#Applying to Blue channel and then applying inverse transform to transformed array.
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)
img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)
imf=Image.fromarray(img_compressed)
imf.show()