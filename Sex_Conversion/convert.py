import cv2
import numpy as np
img = cv2.imread('YY.jpg')
img = cv2.resize(img,(256,256))
cv2.imshow('1',img)
cv2.waitKey(0)
cv2.imwrite('YY.jpg',img)

