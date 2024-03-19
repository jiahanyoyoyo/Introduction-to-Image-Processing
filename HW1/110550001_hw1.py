'''
Author: jiahanyoyoyo harry620325@gmail.com
Date: 2024-03-08 23:51:17
LastEditors: jiahanyoyoyo harry620325@gmail.com
LastEditTime: 2024-03-20 04:02:30
FilePath: \HW1\110550001_hw1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import math

def weights(x):
    x = abs(x)
    if x <= 1:
        return 1-2*(x**2)+(x**3)
    elif x < 2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0
    
def bicubic_itp(img,dst_h,dst_w):
    src_h,src_w,_ = img.shape
    new_size = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            src_x = i*(src_h/dst_h)
            src_y = j*(src_w/dst_w)
            x = math.floor(src_x)
            y = math.floor(src_y)
            u = src_x-x
            v = src_y-y
            tmp = 0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii < 0 or y+jj < 0 or x+ii >= src_h or y+jj >= src_w:
                        continue
                    tmp += img[x+ii,y+jj]*weights(ii-u)*weights(jj-v)
            new_size[i,j] = np.clip(tmp,0,255)
    return new_size

image = cv2.imread("building.jpg")

height, width = image.shape[:2]  # to get size and center of image
center_x = width // 2
center_y = height // 2

NNB1 = np.zeros((height, width,3),dtype=np.uint8)
Bilinear1 = np.zeros((height, width,3),dtype=np.uint8)
Bicubic1 = np.zeros((height, width,3),dtype=np.uint8)
NNB2 = np.zeros((2*height, 2*width,3),dtype=np.uint8)
Bilinear2 = np.zeros((2*height, 2*width,3),dtype=np.uint8)
Bicubic2 = np.zeros((2*height, 2*width,3),dtype=np.uint8)

theta = np.deg2rad(-30)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

for y in range(height):
    for x in range(width):
        new_x = x - center_x
        new_y = y - center_y
        
        pos_x = (cos_theta * new_x - sin_theta * new_y + center_x)
        pos_y = (sin_theta * new_x + cos_theta * new_y + center_y)
        x1 = int(math.floor(pos_x))
        x2 = x1 + 1
        y1 = int(math.floor(pos_y))
        y2 = y1 + 1
        
        
        if (0 <= pos_y <= height - 1) and (0 <= pos_x <= width - 1):
            # Nearest Neighbor Interpolation
            NNB1[y,x] = image[round(pos_y),round(pos_x)]
            
            # Bilinear Interpolation

            Bilinear1[y,x] = (y2 - pos_y) * ((x2 - pos_x)*image[y1,x1] + (pos_x - x1)*image[y1,x2]) + (pos_y - y1) * ((x2 - pos_x)*image[y2,x1] + (pos_x - x1)*image[y2,x2])
            
            # Bicubic Interpolation 
            u = pos_y - y1
            v = pos_x - x1
            tmp = 0
            for ii in range(-1,3):
                for jj in range(-1,3):
                    if y1+ii < 0 or x1+jj < 0 or y1+ii >= height or x1+jj >= width:
                        continue
                    tmp += image[y1+ii,x1+jj]*weights(ii-u)*weights(jj-v)
            Bicubic1[y,x] = np.clip(tmp,0,255)


for y in range(2*height):
    for x in range(2*width):
        # Nearest neighbors Interpolation
        
        NNB2[y,x] = image[y//2,x//2]
        
# *Bilinear Interpolation
for y in range(2*height):
    for x in range(2*width):        
        
        yy = y / 2
        xx = x / 2
        x1 = x // 2
        y1 = y // 2
        x2 = x1 + 1
        y2 = y1 + 1
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        x2 = min(x2, width - 1)
        y2 = min(y2, height - 1)
        Bilinear2[y,x] = (y2 - yy) * ((x2 - xx)*image[y1,x1] + (xx - x1)*image[y1,x2]) + (yy - y1) * ((x2 - xx)*image[y2,x1] + (xx - x1)*image[y2,x2])

# *Bicubic interpolation
Bicubic2 = bicubic_itp(image, 2* height, 2* width)


cv2.imwrite("NNB1.jpg", NNB1)
cv2.imwrite("Bilinear1.jpg", Bilinear1)
cv2.imwrite("Bicubic1.jpg", Bicubic1)
cv2.imwrite("NNB2.jpg", NNB2)
cv2.imwrite("Bilinear2.jpg", Bilinear2)
cv2.imwrite("Bicubic2.jpg", Bicubic2)


# img = cv2.resize(image,(1280, 1280), interpolation=cv2.INTER_CUBIC)
# cv2.imshow("img",img)

cv2.waitKey(0)
cv2.destroyAllWindows()


