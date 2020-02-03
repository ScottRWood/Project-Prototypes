import cv2
import numpy as np

image = cv2.imread('pitch-mask-test.jpg')
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('Original Image', image)
cv2.imshow('HSV Image', hsvImage)

cv2.waitKey(0)
cv2.destroyAllWindows()