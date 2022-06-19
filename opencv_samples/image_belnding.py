import cv2
import numpy as np

dog1 = cv2.imread("./images/golang.jpg")
dog2 = cv2.imread("./images/golang.jpg")

print(dog1.shape)
print(dog2.shape)

dog1_dog2 = np.hstack((dog1[:, :256], dog2[: ,256:]))

# generate Gaussian pyramid for dog1

dog1_copy = dog1.copy()
gp_dog1 = [dog1_copy]

for i in range(6):
    dog1_copy = cv2.pyrDown(dog1_copy)
    gp_dog1.append(dog1_copy)

# generate Gaussian pyramid for dog2

dog2_copy = dog2.copy()
gp_dog2 = [dog2_copy]

for i in range(6):
    dog2_copy = cv2.pyrDown(dog2_copy)
    gp_dog2.append(dog2_copy)

# generate laplacian pyramid for dog1

dog1_copy = gp_dog1[5]

# cv2.imshow("dog1", dog1)
# cv2.imshow("dog2", dog2)
cv2.imshow("dog1_dog2", dog1_dog2)

cv2.waitKey(0)
cv2.destroyAllWindows()