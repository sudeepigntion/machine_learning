import cv2

img = cv2.imread("./images/golang.jpg")
img_water_mark = cv2.imread("./images/sudeeps_dog.png")

print(img.shape) # returns a tuple of number of rows, columns and channels
print(img.size) # returns total number of pixels is accessed
print(img.dtype) # returns image datatype is obtained

b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

# ball = img[280: 340, 330:390]

# img[280: 340, 330:390] = ball

img = cv2.resize(img, (512, 512))

img_water_mark = cv2.resize(img_water_mark, (512, 512))

# dst = cv2.add(img, img_water_mark)

dst = cv2.addWeighted(img, .9, img_water_mark, .1, 0)

cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()