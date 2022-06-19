import cv2

img = cv2.imread("./images/dog.11.jpg", -1) # 0, 1, -1 -> specifies the channel of image, (color, gray scale, alpha channel)

print(img)

cv2.imshow('image', img)

k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("./images/sudeeps_dog.png", img)
    cv2.destroyAllWindows()

# cv2.destroyWindow()
