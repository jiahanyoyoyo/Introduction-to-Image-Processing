'''
Author: jiahanyoyoyo harry620325@gmail.com
Date: 2024-04-01 15:09:49
LastEditors: jiahanyoyoyo harry620325@gmail.com
LastEditTime: 2024-04-01 19:52:53
FilePath: \HW2\Histogram_Specification.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Q2_source.jpg",cv2.IMREAD_GRAYSCALE)
ref = cv2.imread("Q2_reference.jpg",cv2.IMREAD_GRAYSCALE)


img_h,img_w = img.shape
ref_h,ref_w = ref.shape

img_total = img_h * img_w
ref_total = ref_h * ref_w

plt.hist(img.ravel(),256,[0,256])

img_cnt = np.zeros(256,dtype=np.int32)
ref_cnt = np.zeros(256,dtype=np.int32)

for i in range(img_h):
    for j in range(img_w):
        img_cnt[img[i,j]] += 1

for i in range(ref_h):
    for j in range(ref_w):
        ref_cnt[ref[i,j]] += 1
        
        
img_probilities = np.zeros(256,dtype=np.float32)
ref_probilities = np.zeros(256,dtype=np.float32)


for i in range(256):
    img_probilities[i] = img_cnt[i] / img_total
    ref_probilities[i] = ref_cnt[i] / ref_total
    
img_cdf = np.zeros(256,dtype=np.float32)
ref_cdf = np.zeros(256,dtype=np.float32)

img_cdf[0] = img_probilities[0]
ref_cdf[0] = ref_probilities[0]

for i in range(1,256):
    img_cdf[i] = img_cdf[i - 1] + img_probilities[i]
    ref_cdf[i] = ref_cdf[i - 1] + ref_probilities[i]
    
img_map = np.zeros(256,dtype=np.uint8)

for i in range(256):
    min_diff = 1.0
    for j in range(256):
        diff = abs(img_cdf[i] - ref_cdf[j])
        if diff < min_diff:
            min_diff = diff
            img_map[i] = j

for i in range(img_h):
    for j in range(img_w):
        img[i,j] = img_map[img[i,j]]
        
plt.hist(img.ravel(), 256,[0,256])
plt.hist(ref.ravel(), 256,[0,256])

plt.legend(('before matching','after matching','target'),loc = 'upper left')
plt.xlabel('Intensity')
plt.ylabel('# of pixels')

plt.savefig('hist_spec.png',dpi = 300,bbox_inches ='tight')
plt.show()

cv2.imwrite('Q2_spec.jpg',img)