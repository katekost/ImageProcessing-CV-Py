import cv2
import numpy as np

im = cv2.imread('init.jpg')
cv2.imread('init.jpg', cv2.IMREAD_COLOR).dtype
h,w = im.shape[:2]
print(h,w)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imwrite('result_to_gray_scale.jpg', gray)

# Compute integral image

intim = cv2.integral(gray)

# Normalize and save

intim = (255.0*intim) / intim.max()
cv2.imwrite('result_integral.jpg',intim)
rgb_im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
cv2.imwrite('result_to_RGB.jpg', rgb_im)

# Convert to signed 16 bit. This will allow values less than zero and greater than 255

img = np.int16(im)
contrast   = 64
brightness = 0
img = img*(contrast/127 + 1) - contrast + brightness
img = np.clip(img, 0, 255)
img = np.uint8(img)
cv2.imwrite('mandrill_contrast64.png', img)

#-----------------------------------------------------------------------------------

im = cv2.imread('init.jpg')
gray1 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imwrite('test.jpg', gray1)

gray1.min() #13
gray1.max() #240
gray1.mean() #128

def piecewise_linear(x, u1, u2, G):
    if (x < u1):
        return 0
    elif (u1 <= x <= u2):
        return ((x-u1)/(u2-u1))*(G-1)
    else:
        return (G-1)

def piecewise_linear2(x, u1, u2, G):
    if (x < u1):
        return 0
    elif (u1 <= x <= u2):
        return ((x-u1)/(u2-u1))*(G-1)
    else:
        return (G-240)

i=0
j=0
result=[]
for i in range(gray1.shape[0]):
    for j in range(gray1.shape[1]):
        gray1[i][j]=piecewise_linear(gray1[i][j], 90, 210, 129)


im = cv2.imread('init.jpg')
gray2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imwrite('init.jpg', gray2)

gray2.min() #13
gray2.max() #240
gray2.mean() #128

for i in range(gray1.shape[0]):
    for j in range(gray2.shape[1]):
        gray1[i][j]=piecewise_linear(gray2[i][j], 90, 210, 129)

cv2.imwrite('test2.jpg', gray2)
