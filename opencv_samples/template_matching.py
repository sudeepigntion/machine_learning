import cv2
import numpy as np

img = cv2.imread("./images/pic.jpg")
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_img = cv2.imread("./images/sudeep_face.JPG", 0)

res = cv2.matchTemplate(grey_img, face_img, cv2.TM_CCOEFF_NORMED)

w, h = face_img.shape[::-1] # reverse order

print(res)

threshold = 0.86

loc = np.where(res >= threshold)

print(loc)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow("face matching", img)

cv2.waitKey(0)
cv2.destroyAllWindows()