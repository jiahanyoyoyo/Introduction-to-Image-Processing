import numpy as np
import cv2


laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    output_image = np.zeros_like(image, dtype=np.float32)  # Use float type for proper normalization
    
    for y in range(image_height):
        for x in range(image_width):
            patch = padded_image[y:y+kernel_height, x:x+kernel_width]
            output_image[y, x] = np.sum(patch * kernel)
    
    # Normalize the output to match cv2.filter2D
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image


image = cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)

result = convolve(image,laplacian_kernel)
# result = cv2.filter2D(image,-1,laplacian_kernel)

cv2.imshow("convolve", result)
cv2.imshow("original",image)
cv2.imwrite('spatial.jpg',result)

cv2.waitKey(0)
cv2.destroyAllWindows()
