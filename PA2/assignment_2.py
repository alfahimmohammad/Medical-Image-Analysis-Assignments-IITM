# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:25:41 2021

@author: alfah
"""
#%%
#importing required libraries
#I have renamed the images as '1.png' and '2.png' for convenience
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#%%
#Question 1 
def gaussnoise(image,sigma=1):
    row,col= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col))
    #gauss = gauss.reshape(row,col)
    noise = image + gauss
    return noise

def gaussfilter(image, size=3, sigma = 1): #defined gaussian function for gaussian filtering
    row, col = image.shape
    #mean = 0
    k = 1/(2*np.pi*sigma**2)
    c = size//2
    filt = np.zeros((size,size),dtype = 'float32')
    for i in range(size):
        for j in range(size):
            f = -((i-c)**2+(j-c)**2)/(2*sigma**2)
            filt[i,j] = k*np.exp(f)
    result = signal.convolve2d(image,filt,mode='same',boundary='symm')
    
    return result

def filterr(img, filt = 'mean'):#filtering function, used this only for filtering, not the above one
    shape = img.shape
    if filt == 'mean':
        lpf = (1/25)*np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    elif filt == 'gauss':
        lpf = (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
    else:#median filtering. Padding 1 layer around the image in 'reflect' mode.
        new_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        for i in range(1,shape[0]+1):
            for j in range(1, shape[1]+1):    
                w = new_img[i-1:i+2,j-1:j+2].ravel()
                new_img[i,j] = np.median(w)
        n_img = new_img[1:shape[0]+1,1:shape[1]+1]
        return np.uint8(n_img)
    result = signal.convolve2d(img,lpf,mode='same',boundary='symm')
    return np.uint8(result)
        
img = cv2.imread('60/2.jpg',0)
noisy_img = gaussnoise(img, sigma=5)#3 for 1; 5 for 2
mean = filterr(noisy_img)
gauss = filterr(noisy_img,'gauss')
median = filterr(noisy_img,'median')
noisy_img=cv2.normalize(src=noisy_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print('mean: ',cv2.PSNR(img,mean))
print('gauss: ',cv2.PSNR(img,gauss))
print('median: ',cv2.PSNR(img,median))
print('noise: ',cv2.PSNR(img,noisy_img))

#%%
#plotting the results of Question 1
plt.imshow(img,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('original img 2')
plt.subplot(2,2,2)
plt.imshow(mean,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('mean filtered')
plt.subplot(2,2,3)
plt.imshow(gauss,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('gauss filtered')
plt.subplot(2,2,4)
plt.imshow(median,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('median filtered')
plt.imshow(noisy_img,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('gaussian noise std:5')
plt.imshow(mean-img,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('mean residual')
plt.imshow(gauss-img,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('gaussian residual')
plt.imshow(median-img,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('median residual')
#%%
#Contrast enhanement for Question 2
def gamma(im,g):
    # new r = c*r^(gamma) ; c = 1
    imag = np.zeros(im.shape)
    imag = im**g
    return imag

img = cv2.imread('60/2.jpg',0)
equ = cv2.equalizeHist(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25,25))#25,25 for 1; 50,50 for 2
cl1 = clahe.apply(img)
g = gamma(img,1.9) #0.5 for 1, 1.9 for 2
g = cv2.normalize(src=g, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.imshow(g,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('gamma=1.9')
#%%
#plotting histogram for performance comparison
plt.hist([img.ravel(),equ.ravel(),cl1.ravel(),g.ravel()],256,[0,256]);plt.legend(['Original','HE','CLAHE','gamma=0.5']);plt.xlabel('Pixel intensity');plt.ylabel('No. of pixels');plt.title('image 1')
#%%
#Edge detection for Question 2
#Below are the indices for sternum cropping
#180:240,415:540 for img 1
#140:200,387:500 for img 2
def edge(im,etype='sobel'):
    if etype == 'sobel':  
        vedge = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
        hedge = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    elif etype == 'prewitt':
        vedge = np.asarray([[-1,-1,-1],[0,0,0],[1,1,1]])
        hedge = np.asarray([[-1,0,1],[-1,0,1],[-1,0,1]])
    else:
        lapmask = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        edge = signal.convolve2d(im,lapmask,mode='same',boundary='symm')
        return cv2.normalize(src=edge, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),1,2
    gx = signal.convolve2d(im,hedge,mode='same',boundary='symm')
    gy = signal.convolve2d(im,vedge,mode='same',boundary='symm')
    G = np.absolute(gx) + np.absolute(gy)
    return cv2.normalize(src=G, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), np.absolute(gx), np.absolute(gy)

f =cv2.GaussianBlur(g[140:200,387:500], ksize=(5,5), sigmaX=3, borderType=cv2.BORDER_REPLICATE)
#Canny= (5,5), 5 for 1 ;(5,5), 3 for 2
#%%
#plotting results of edge detection
S,_,_ = edge(f,'lapmask')
plt.imshow(S,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('Laplacian of Gaussian')
#%%
#plotting results of canny edge detection
canny = cv2.Canny(f,70,100)#70,100 for 1; 70,100 for 2
_,g = cv2.threshold(np.uint8(S), 90, 255, cv2.THRESH_BINARY)
plt.imshow(canny,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('Canny')
#%%
#Hough transform for Question 3

X, Y = canny.shape
ab = np.zeros((2*360,2*360))
n=4
t = np.linspace(0, 2 * np.pi, 360)
#I have followed the 2nd set equations as shown in the answer of Question 3 in the report 
#The following loop takes care of voting in the parameter space
for i in range(X):
    for j in range(Y):
        if canny[i,j] == 255:
            for T in t:
                if np.sign(np.cos(T))==1.0 and np.sign(np.sin(T))==1.0 and int(i/(((np.abs(np.cos(T))) ** (2 / n)) * np.sign(np.cos(T)) + 1e-06)) < 720 and int(j/(((np.abs(np.sin(T))) ** (2 / n)) * np.sign(np.sin(T)) + 1e-06)) < 720:
                    ab[int(i/(((np.abs(np.cos(T))) ** (2 / n)) * np.sign(np.cos(T)))), 
                       int(j/(((np.abs(np.sin(T))) ** (2 / n)) * np.sign(np.sin(T))))] += 1

#center is (60,30) for 1, (55,30) for 2
#the following loop thresholds the votes in the parameter space
ab = ab/np.max(ab)
index = np.zeros((1,2))
for i in range(ab.shape[0]):
    for j in range(ab.shape[1]):
        if ab[i,j] >0.7:# and ab[i,j]<0.8: #0.8 for 1, 0.7 for 2
            index = np.concatenate((index, np.array([i,j],ndmin=2)),axis=0)

n=4 #3 for both 1 and 2
#The final loop plots the ellipses using the a, b values obtained from the above loop
t = np.linspace(0, 2 * np.pi, 360)
for i in range(index.shape[0]-1):
    a, b = index[i+1,:]
    x = 55 + ((np.abs(np.cos(t))) ** (2 / n)) * a * np.sign(np.cos(t))
    y = 30 + ((np.abs(np.sin(t))) ** (2 / n)) * b * np.sign(np.sin(t))
    plt.plot(x,y)
plt.imshow(canny,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('Hough transform(m=4) of sternum 2')
plt.show()
    

#useful tip: np.unravel_index(np.argmax(a, axis=None), a.shape) : for unraveling argmax and argmin

