import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

img=cv2.imread('train-tif/train_10051.tif',cv2.IMREAD_UNCHANGED)
print(img.shape)
print(img.dtype)
r,g,b,i=cv2.split(img)
print(np.max(r))
print(np.min(r))
print(np.max(g))
print(np.min(g))
print(np.max(b))
print(np.min(b))
print(np.max(i))
print(np.min(i))
rgb=cv2.merge((b,g,r))

img2=cv2.imread('train-jpg-sample/train_10051.jpg',cv2.IMREAD_UNCHANGED)
b2,g2,r2=cv2.split(img2)
print(img2.dtype)
print(np.max(r2))
print(np.min(r2))
print(np.max(g2))
print(np.min(g2))
print(np.max(b2))
print(np.min(b2))

img3 = io.imread('train-tif/train_10051.tif')
r3,g3,b3,i3=img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3]
print(np.max(r3))
print(np.min(r3))
print(np.max(g3))
print(np.min(g3))
print(np.max(b3))
print(np.min(b3))
print(np.max(i3))
print(np.min(i3))
#plt.imshow(r3)
cv2.imshow('skimage',cv2.merge((b3,g3,r3)))

cv2.imshow('test',rgb)
cv2.imshow('test2',i)
cv2.imshow('jpeg',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
