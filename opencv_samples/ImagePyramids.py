import cv2
import numpy as np

img = cv2.imread("./images/golang.jpg")

layer = img.copy()

gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    #cv2.imshow(str(i), layer)

# lr1 = cv2.pyrDown(img)
# lr2 = cv2.pyrDown(lr1)

# hr1 = cv2.pyrUp(lr2)
# hr2 = cv2.pyrUp(hr1)

# cv2.imshow("Original Image", img)
# cv2.imshow("pyrDown 1 Image", lr1)
# cv2.imshow("pyrDown 2 Image", lr2)

# cv2.imshow("pyrUp 1 Image", hr1)
# cv2.imshow("pyrUp 2 Image", hr2)

layer = gp[5]

cv2.imshow("upper level Gaussian Pyramid", layer)

lp = [layer]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gp[i])
    lalacian = cv2.subtract(gp[i-1], gaussian_extended)
    cv2.imshow(str(i), lalacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
