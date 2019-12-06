import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('C:/Users/Arystan/PycharmProjects/MLrestaurant/origin.jpg', 0)
plt.imshow(img)

height, width = img.shape[:2]

img_resize = cv2.resize(img, (365, 56))
plt.imshow(img_resize)
(h, w) = img.shape[:2]
center = (w / 2, h / 2)
angle90 = 90
angle180 = 180
angle270 = 270
scale = 1.0

matrix180 = cv2.getRotationMatrix2D(center, angle180, scale)
rotation180 = cv2.warpAffine(img, matrix180, (width, height))
plt.imshow(rotation180)

(rows, columns) = img.shape
pts1 = np.float32([[50, 50], [200, 100], [10, 150]])
pts2 = np.float32([[10, 10], [200, 100], [0, 100]])
Mat = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, Mat, (columns, rows))
plt.subplot(121)
plt.imshow(img)
plt.title('Before')
plt.subplot(122)
plt.imshow(dst)
plt.title('After')
plt.show()

img = cv2.medianBlur(img, 5)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite('C:/Users/Айгерим/Desktop/s600.jpg', img)
cv2.imwrite('C:/Users/Айгерим/Desktop/s600.jpg', th1)
cv2.imwrite('C:/Users/Айгерим/Desktop/s600.jpg', th2)
cv2.imwrite('C:/Users/Айгерим/Desktop/s600.jpg', th3)
cv2.imwrite('C:/Users/Айгерим/Desktop/s600.jpg', rotation180)
