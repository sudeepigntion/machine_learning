import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./images/golang.jpg", 0)

canny = cv2.Canny(img, 100, 200)

titles = ["image", "canny"]
images = [img, canny]

# The cannt edge detection algorithm is composed of 5 steps:
# 1. Noise reduction
# 2. Gradient calculation
# 3. Non-maximum suppression
# 4. Double Threshold
# 5. Edge Tracking by Hysteresis

for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show() 