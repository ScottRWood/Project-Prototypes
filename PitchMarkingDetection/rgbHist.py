import cv2
import numpy as np

orig = cv2.imread('pitch-mask-test.jpg')
src = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

hsv_planes = cv2.split(src)
histSize = 256
histRange = (0,256)

accumulate = False

h_hist = cv2.calcHist(hsv_planes, [0], None, [histSize], histRange, accumulate=accumulate)
s_hist = cv2.calcHist(hsv_planes, [1], None, [histSize], histRange, accumulate=accumulate)
v_hist = cv2.calcHist(hsv_planes, [2], None, [histSize], histRange, accumulate=accumulate)

hist_w = 512
hist_h = 400
bin_w = int(round(hist_w/histSize))

histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

cv2.normalize(h_hist, h_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(s_hist, s_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(v_hist, v_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

for i in range(1, histSize):
    cv2.line(histImage, ( bin_w*(i-1), hist_h - int(round(h_hist[i-1][0])) ), ( bin_w*(i), hist_h - int(round(h_hist[i][0])) ), ( 255, 0, 0), thickness=2)
    cv2.line(histImage, ( bin_w*(i-1), hist_h - int(round(s_hist[i-1][0])) ), ( bin_w*(i), hist_h - int(round(s_hist[i][0])) ), ( 0, 255, 0), thickness=2)
    cv2.line(histImage, ( bin_w*(i-1), hist_h - int(round(v_hist[i-1][0])) ), ( bin_w*(i), hist_h - int(round(v_hist[i][0])) ), ( 0, 0, 255), thickness=2)

min_max_bin = cv2.minMaxLoc(h_hist)
mask = None
cv2.inRange(src, min_max_bin[1], min_max_bin[1], mask)

cv2

print(min_max_bin)

cv2.imshow('Source', src)
cv2.imshow('Hist', histImage)
cv2.waitKey()