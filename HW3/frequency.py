import numpy as np
import cv2

laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

image = cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)

spectrum = np.fft.fft2(image)

padded_kernel = np.zeros_like(image, dtype=np.float64)
padded_kernel[:laplacian_kernel.shape[0], :laplacian_kernel.shape[1]] = laplacian_kernel

padded_kernel_spectrum = np.fft.fft2(padded_kernel)

filtered_spectrum = spectrum * padded_kernel_spectrum

filtered_image = np.fft.ifft2(filtered_spectrum).real

output_image = np.clip(filtered_image, 0, 255).astype(np.uint8)


cv2.imshow("original",image)
cv2.imshow("filtered",output_image)
cv2.imwrite("frequency.jpg",output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()