import numpy as np
import cv2 as cv

def nothing(x):
    print(x)

# Create a black image, a window

# img = np.zeros((300, 512, 3), np.uint8)

cv.namedWindow("image")

cv.createTrackbar("CP", "image", 10, 400, nothing)

# cv.createTrackbar("B", "image", 10, 400, nothing)
# cv.createTrackbar("G", "image", 0, 255, nothing)
# cv.createTrackbar("R", "image", 0, 255, nothing)

switch = 'color/gray'

cv.createTrackbar(switch, 'image', 0, 1, nothing)

while True:
    img = cv.imread("./images/golang.jpg")
    pos = cv.getTrackbarPos("CP", "image")
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, str(pos), (50, 150), font, 4, (0, 0, 255))


    k = cv.waitKey(1) & 0xFF
    if k ==27:
        break

    s = cv.getTrackbarPos(switch, "image")

    # b = cv.getTrackbarPos("B", "image")
    # g = cv.getTrackbarPos("G", "image")
    # r = cv.getTrackbarPos("R", "image")
    # s = cv.getTrackbarPos(switch, "image")

    if s == 0:
        pass
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = cv.imshow("image", img)

cb.destroyAllWindows()