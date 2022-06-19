import cv2
import numpy as np

#img = cv2.imread("./images/golang.jpg", -1) # 0, 1, -1 -> specifies the channel of image, (color, gray scale, alpha channel)

img = np.zeros([512, 512, 3], np.uint8)

# image, start cordinates, end cordinates, bgr color, thickness, 

img = cv2.line(img, (0, 0), (255, 255), (147, 96, 44), 10)

img = cv2.arrowedLine(img, (0, 255), (255, 255), (255, 0, 0), 10)

img  = cv2.rectangle(img, (384, 0), (510, 128), (0, 0, 255), 5) # -1 will fill with color

img  = cv2.circle(img, (447, 63), 63, (0, 255, 0), -1)

img = cv2.putText(img, "OpenCV", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA)

print(img)

cv2.imshow('image', img)

k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("./images/sudeeps_dog.png", img)
    cv2.destroyAllWindows()

# cv2.destroyWindow()
