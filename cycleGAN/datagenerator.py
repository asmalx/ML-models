import keras
import numpy as np
import math


import skimage.morphology as skmorf
from scipy import signal
from scipy.ndimage import gaussian_filter
from skimage.draw import line
from scipy.interpolate import RegularGridInterpolator
import skimage


import os
from matplotlib import image as Image
import keras

import cv2

import augmentation

'''
For tests: X1 is always ones, X2 is always zeros

'''
class TestGenerator(keras.utils.Sequence):
    def __init__(self, shape, batch_size=1, len=1000):
        self.len = len
        self.batch_size = batch_size
        self.sample_shape = shape
        self.shape = (batch_size, *shape)


    def __getitem__(self, index):
        x1, x2 = np.ones((self.shape)), np.zeros((self.shape))
        return (x1, x2)

    def __len__(self):
        return self.len 


class TestMNISTGeneator(keras.utils.Sequence):
    def __init__(self,  x1_lbl : int, x2_lbl : int, augmentation=augmentation.AugmentationUnit()):
        (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()
        xtrain = xtrain.astype('float')
        xtrain = xtrain / 255.0
        self.x1 = xtrain[ytrain==x1_lbl]
        self.x2 = xtrain[ytrain==x2_lbl]
        self.len = min(self.x1.shape[0], self.x2.shape[0])
        self.aug = augmentation

    def __getitem__(self, index):
        x1 = np.repeat(self.x1[index].reshape((28,28,1)), 3, axis=-1)
        x2 = np.repeat(self.x2[index].reshape((28,28,1)), 3, axis=-1)
        return (np.mean(self.aug(x1), axis=-1)[np.newaxis, :,:,  np.newaxis],  np.mean(self.aug(x2), axis=-1)[np.newaxis, :,:,  np.newaxis])

    def __len__(self):
        return self.len     

    def on_epoch_end(self):
        np.random.shuffle(self.x1)     
        np.random.shuffle(self.x2)           


# batch size is fixed and eq to 1
class CatDogsCIFARGenerator(keras.utils.Sequence):
    def __init__(self,  augmentation: bool):
        (xtrain, ytrain), (xtest, ytest) = keras.datasets.cifar10.load_data()
        self.vx1, self.vx2 =  xtest[ytest[:, 0]==3],  xtest[ytest[:, 0]==5]
        self.x1data, self.x2data = xtrain[ytrain[:, 0]==3],  xtrain[ytrain[:, 0]==5]
        self.x1data, self.x2data = self.x1data / 255., self.x2data / 255.
        self.vx1, self.vx2 = self.vx1 / 255., self.vx2 / 255.

        del xtrain, ytrain, xtest, ytest
  
        self.on_epoch_end()

        self.len = min(self.x1data.shape[0], self.x2data.shape[0])

    def __getitem__(self, index):
        return (self.x1data[index].reshape(1, 32, 32, 3),  self.x2data[index].reshape(1, 32, 32, 3),)

    def get_val_item(self, index):
         return (self.vx1[index].reshape(1, 32, 32, 3),  self.vx2[index].reshape(1, 32, 32, 3),)
       

    def __len__(self):
        return self.len     

    def on_epoch_end(self):
        np.random.shuffle(self.x1data)     
        np.random.shuffle(self.x2data) 





#cv2.BORDER_REFLECT
#cv2.BORDER_REPLICATE

# Batch_size = 1



class ImgFileIterator(keras.utils.Sequence):
    def __init__(self, path_x1, path_x2, out_shape3d,  augmentation=augmentation.AugmentationUnit(), fill_mode='nearest', preprocessing_f = None):
        self.shape = out_shape3d
        self.root_ps1 = path_x1
        self.root_ps2 = path_x2
        self.ps1 = os.listdir(path_x1) 
        self.ps2 = os.listdir(path_x2)
        self.aug = augmentation
        self.fill = cv2.BORDER_REPLICATE if fill_mode=='nearest' else cv2.BORDER_REFLECT
        self.pf = preprocessing_f


    def __pad_to_square(self, img):
        h,w, _ = img.shape
        m = max(h, w)
        image = cv2.copyMakeBorder(img,(m-h)//2,(m-h)//2,(m-w)//2,(m-w)//2, self.fill)
        return image 
    
    def __getitem__(self, idx):
        #load
        img1 = Image.imread(self.root_ps1 + '/' + self.ps1[idx])
        img2 = Image.imread(self.root_ps2 + '/' + self.ps2[idx])
        # norm
        img1, img2 = img1 - img1.min(), img2 - img2.min()
        img1, img2 = img1 / img1.max(),  img2 / img2.max()     
        # pad to square
        img1 = self.__pad_to_square(img1)
        img2 = self.__pad_to_square(img2)

        # preprocess for faster augmentation
        img1 = cv2.resize(img1, 2*np.array(self.shape[:-1]))
        img2 = cv2.resize(img2, 2*np.array(self.shape[:-1]))

        # augmentate
        img1 = self.aug(img1)
        img2 = self.aug(img2)
        #resize
        img1 = cv2.resize(img1, self.shape[:-1])
        img2 = cv2.resize(img2, self.shape[:-1])
        
        if self.pf != None:
            img1 = self.pf(img1)
            img2 = self.pf(img2)
            return (img1, img2)
        return (img1.reshape((1,*self.shape)),  img2.reshape((1,*self.shape)))
        
 
    def __len__(self):
        return min(len(self.ps1), len(self.ps2))

    def on_epoch_end(self):
        np.random.shuffle(self.ps1)     
        np.random.shuffle(self.ps2) 






class AugmentationSequence(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        # Initialization
        self.shuffle = shuffle
        self.batch_size = batch_size
        if shuffle:
            self.i = np.array(range(x.shape[0]))
            np.random.shuffle(self.i)
            self.x, self.y = x[self.i], y[self.i]
        else:
            self.x = x
            self.y = y
        
        self.size = len(y)

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.i[index*self.batch_size:(index+1)*self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        
        # preprocess data
        for i in range(self.batch_size):
            x_batch[i] = augmentation_pipeline(x_batch[i])
        return x_batch, y_batch
    
    def __len__(self):
        #  batches per epoch
        return self.size // self.batch_size

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.i)        
        pass
from collections import deque






# input should be 2-dimensional grayscale image
def linear_shift(img, horizontal_shift_px, vertical_shift_px, fill=0.):
    H, W = img.shape
    h, v = horizontal_shift_px, vertical_shift_px
    out = fill*np.ones((H,W))
    out[max(0,-v):H-v, max(0,-h):W-h] = img[max(0,v):min(H,H+v), max(0,h):min(W, W+h)]
    return out


def zoom(img, zoom=1., fill=0.):
    out = fill*np.ones(img.shape)
    true_zoomed = scipy.ndimage.zoom(img, zoom=zoom, order=3, mode='constant', cval=fill)
    X, Y = img.shape
    x, y = true_zoomed.shape
    if x > X:
        out = true_zoomed[math.floor(x/2-X/2):math.floor(X/2-x/2), math.floor(y/2-Y/2):math.floor(Y/2-y/2)]
    elif x < X:
        out[math.floor(X/2-x/2):math.floor(x/2-X/2), math.floor(Y/2-y/2):math.floor(y/2-Y/2)] = true_zoomed
    else: out = true_zoomed
    return out 

def rotate(img, angle, fill=0.):
    return scipy.ndimage.rotate(img, angle, reshape=False, cval=fill)




def contrast_brightness_adj(img, brightness, contrast):
    # brightness
    img = img + brightness
    # contrast
    img = contrast*(img - 0.5) + 0.5
    return np.clip(img, 0., 1.)