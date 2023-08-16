# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 08:59:49 2021

@author: alfah
"""

#%%
#Q1
import cv2
import glob
import numpy as np
import maxflow
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import color 
#I renamed the files in my folder as *_x for images and *_y for labels,just for easy reading

#Reading images and labels
images = [cv2.imread(file,0) for file in glob.glob("52/*_x.bmp")]
labels = [cv2.imread(file,0) for file in glob.glob("52/*_y.bmp")]

#Thresholding labels for binary segmentation
for item in labels:
    item[item>0] = 1
    
beta_intens = 0.5
beta_freq = 0.5
#1.3 for 0,1; 1.1 for 2, 1 for 3, 0.5 for 4

ppv = []

def pitter(peak, pit, ppv, hist):
    if pit == len(hist)-1:
        if hist[pit]<hist[pit-1]:
            ppv.append(pit)
            return ppv
        else:
            return ppv
    elif hist[pit-1]>hist[pit] and hist[pit]<=hist[pit+1]:
        ppv.append(pit)
        peak = pit + 1
        return peaker(peak, pit, ppv, hist)
    else:
        pit += 1
        return pitter(peak, pit, ppv, hist)

def peaker(peak, pit, ppv, hist):
    if peak ==0 and hist[peak]>hist[peak+1]:
        ppv.append(peak)
        pit = peak + 1
        return pitter(peak, pit, ppv, hist)
    elif peak == len(hist)-1:
        return ppv
    elif hist[peak-1]<=hist[peak] and hist[peak]>hist[peak+1]:
        ppv.append(peak)
        pit = peak + 1
        return pitter(peak, pit, ppv, hist)
    else:
        peak += 1
        return peaker(peak, pit, ppv, hist)
    

img = images[4]
hist = np.squeeze(cv2.calcHist([img],[0],None,[256],[0,256]))
shape = img.shape
#generating PeakPitVector
ppv = []
peak = 0
pit = 0
ppv = peaker(peak, pit, ppv, hist)
#intensity search
if len(ppv)%2 == 0:
    ld = []
    i=0
    while i<len(ppv):
        ld.append(abs(ppv[i] - ppv[i+1]))
        i += 2
'''
else:
    print('PPV is not pairwise')
    break
'''
th_intens = np.mean(ld)*beta_intens
sld = []
for i in range(len(ld)):
    if ld[i]>th_intens:
        sld.append(ppv[2*i])
        sld.append(ppv[2*i+1])

#frequency search
if len(sld)%2 == 0:
    hd = []
    i=0
    while i<len(sld):
        hd.append(abs(hist[sld[i]] - hist[sld[i+1]]))
        i += 2
'''
else:
    print('SLD is not pairwise')
    break
'''
th_freq = np.mean(hd)*beta_freq
shd = []
for i in range(len(hd)):
    if hd[i]>th_freq:
        shd.append(sld[2*i])
        shd.append(sld[2*i+1])
    
#number of clusters is len(shd)
shd = np.array(shd,ndmin=2).T

kmeans = KMeans(n_clusters=2)
kmeans.fit(shd)

z = kmeans.predict(shd)

foreground = np.zeros(shape=img.shape, dtype='uint8')
background = np.zeros(shape=img.shape, dtype='uint8')


f = z[0]
b = z[-1]
for i in range(len(z)):    
    if z[i] == f:
        foreground[img==shd[i,0]] = 255
    if z[i] == b:
        background[img==shd[i,0]] = 255
        
alpha = 0.6


# Construct RGB version of grey-level image
color_mask = np.dstack((foreground, np.zeros(shape=img.shape, dtype='uint8'), background))
img_color = np.dstack((img, img, img))

# Convert the input image and color mask to Hue Saturation Value (HSV)
# colorspace
img_hsv = color.rgb2hsv(img_color)
color_mask_hsv = color.rgb2hsv(color_mask)

# Replace the hue and saturation of the original image
# with that of the color mask
img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

img_masked = color.hsv2rgb(img_hsv)

# Display the output
plt.imshow(color_mask);plt.xticks([]);plt.yticks([]);plt.title('Assigned class labels')
#plt.imshow(img,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('Original image 5')


#%%
#Q2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import os

from maxflow.fastmin import abswap_grid, aexpansion_grid


I=img/255

levs = np.squeeze(shd)/255
# Calculate data cost as the absolute difference between the label prototype and the pixel value
D = np.abs(I.reshape(I.shape+(1,)) - levs.reshape((1,1,-1)))

# Generate nearest prototype labeling
Id = np.argmin(D,2)

# Calculate neighbourhood cost as absolute difference between prototypes 
alpha = 1
V = alpha * np.abs(levs.reshape((-1,1)) - levs.reshape((1,-1)))

# Mimimise data + neighbourhood cost
#label = abswap_grid(D,V)
label = aexpansion_grid(D,V)

fg = plt.figure("Regularised labeling")
ax1 = fg.add_subplot(1,1,1)
ax1.imshow(label,cmap='gray');ax1.set_xticks([]);ax1.set_yticks([])
#ax1.set_title('alpha beta swap')
ax1.set_title('alpha expansion')


#below are the intensity values used to set the conditions for segmentation(threshold-based)
#1 for 0,1; 1,2,3 for 2; 0,1,2 for 3; 0,8 for 4;
segment = np.ones(I.shape, dtype='uint8')
segment[label==0] = 0
#segment[label==2] = 0
#segment[label==0] = 0
segment[label==8] = 0
plt.imshow(segment,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('Thresholded result')
#%%
def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def accuracy(result, reference):
    # accuracy=(TP+TN)/(TP+FN+FP+TN)
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    tp=  np.count_nonzero(result & reference) 
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    fn = np.count_nonzero(~result & reference) 
    try:
        accuracy=(tp+tn)/(tp+tn+fp+fn)
    except ZeroDivisionError:
        accuracy = 0.0
    
    return accuracy

def sensitivity(result, reference):
    # accuracy=(TP+TN)/(TP+FN+FP+TN)
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    tp=  np.count_nonzero(result & reference) 
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    fn = np.count_nonzero(~result & reference) 
    try:
        sensitivity=(tp/(tp+fn+1))
    except ZeroDivisionError:
        sensitivity = 0.0
    
    return sensitivity

def specificity(result, reference):
    # accuracy=(TP+TN)/(TP+FN+FP+TN)
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    tp=  np.count_nonzero(result & reference) 
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    fn = np.count_nonzero(~result & reference) 
    try:
        specificity=(tn/(tp+fn+1))
    except ZeroDivisionError:
        specificity = 0.0
    
    return specificity


def jc(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    
    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    
    jc = float(intersection) / float(union) #tp/tp+fn+fp
    
    return jc

true = labels[4]
print('dice: ', dice(segment,true))
print('acc: ', accuracy(segment,true))
print('sens: ', sensitivity(segment,true))
print('spec: ', specificity(segment,true))
print('jacc: ', jc(segment,true))
    
    
            
            
            
            
            
            
            
