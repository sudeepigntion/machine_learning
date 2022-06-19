import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

# fbbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)

# fbbg = cv.bgsegm.createBackgroundSubtractorGMG()

fbbg = cv.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = fbbg.apply(frame)

    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    cv.imshow("Frame", frame)
    cv.imshow("GF Mask Frame", fgmask)

    keyboard = cv.waitKey(30)

    if keyboard == 'q' or keyboard == 27:
        break


cap.release()
cv.destroyAllWindows()