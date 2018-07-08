import os
import sys
import numpy as np
import cv2 as cv

print(os.getcwd())

# SEGMENTATION
# ------------------------------------------------------------------------------
# Watershed Algorithm

img_init = cv.imread("D:/ML/ML/photos/Accessorize.jpg")
img = cv.imread("D:/ML/ML/photos/Accessorize.jpg", 0)
#img_init = cv.imread("D:/ML/ML/photos/teeth.jpg")
#img = cv.imread("D:/ML/ML/photos/teeth.jpg", 0)
#img_init = cv.imread("D:/ML/ML/photos/roadstig.jpg")
#img = cv.imread("D:/ML/ML/photos/roadstig.jpg", 0)
#img_init = cv.imread("D:/ML/ML/photos/cell.jpg")
#img = cv.imread("D:/ML/ML/photos/cell.jpg", 0)
ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(img_init, markers)
img_init[markers == -1] = [255,0,0]
cv.imwrite('waterAccessorize.jpeg', img_init)
#cv.imwrite('waterteeth.jpeg', img_init)
#cv.imwrite('waterroadstig.jpeg', img_init)
#cv.imwrite('watercell.jpeg', img_init)

# ------------------------------------------------------------------------------
# K-Means Algorithm

img = cv.imread("D:/ML/ML/photos/cell.jpg")
#img = cv.imread("D:/ML/ML/photos/roadstig.jpg")
#img = cv.imread("D:/ML/ML/photos/teeth.jpg")
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
#define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#K = 2
K = 3
#K = 5
#K = 8
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imwrite('kcell3.jpeg', res2)
#cv.imwrite('kteeth2.jpeg', res2)
#cv.imwrite('kteeth5.jpeg', res2)
#cv.imwrite('kteeth8.jpeg', res2)
#cv.imwrite('kroadstig2.jpeg', res2)
#cv.imwrite('kroadstig3.jpeg', res2)
#cv.imwrite('kroadstig4
#cv.imwrite('kroadstig5.jpeg', res2)# .jpeg', res2)

# ------------------------------------------------------------------------------
# GrubCut Algorithm

img = cv.imread("D:/ML/ML/photos/road_stig.jpg")
img = cv.imread("D:/ML/ML/photos/cell.jpg")
img = cv.imread("D:/ML/ML/photos/teeth.jpg")
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
cv.imwrite("GrubCutroadstig.jpeg" ,img)
#cv.imwrite("Grubcell.jpeg" ,img)
#cv.imwrite("GrubCutteeth.jpeg" ,img)
