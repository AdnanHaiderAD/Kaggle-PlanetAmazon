import readwrite
import cv2

#label_reader = readwrite.LabelReader('/home/mifs/jhmw2/kaggle/amazon/Kaggle-PlanetAmazon/train.csv')
#print (label_reader.Read('train_1'))

image_reader = readwrite.ImageReader('train.scp')
img = image_reader.Read('train_1')
r,g,b,i=cv2.split(img)
rgb=cv2.merge((b,g,r))
cv2.imshow('test',rgb)
cv2.waitKey()

