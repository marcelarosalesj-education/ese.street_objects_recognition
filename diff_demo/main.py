"""
Image Difference Compare
"""
import sys
import cv2
import numpy as np

scale = 0.5

filename1 = sys.argv[1]
filename2 = sys.argv[2]

img1 = cv2.imread(filename1, 0)
img2 = cv2.imread(filename2, 0)

print("Picture 1 size is {}x{}".format(img1.shape[0], img1.shape[1]))
print("Picture 2 size is {}x{}".format(img2.shape[0], img2.shape[1]))

print("Resizing to {}%".format(scale))

"""
About scaling and resizing:
https://docs.opencv.org/3.4.2/da/d6e/tutorial_py_geometric_transformations.html
"""
rimg1 = cv2.resize(img1, None, fx=scale, fy=scale)
rimg2 = cv2.resize(img2, None, fx=scale, fy=scale)

cv2.imshow("Resized Picture 1", rimg1)
cv2.imshow("Resized Picture 2", rimg2)

print("Resized Picture 1 size is {}x{}".format(rimg1.shape[0], rimg1.shape[1]))
print("Resized Picture 2 size is {}x{}".format(rimg2.shape[0], rimg2.shape[1]))

"""
About image differentiation:
https://stackoverflow.com/questions/51288756/how-do-i-calculate-the-percentage-of-difference-between-two-images-using-python/5128914
https://stackoverflow.com/questions/843972/image-comparison-fast-algorithm
"""

res = cv2.absdiff(rimg1, rimg2)
res = res.astype(np.uint8)
percentage = (np.count_nonzero(res) * 100)/ res.size

print("The percentage is {}".format(percentage))

cv2.waitKey(0)
cv2.destroyAllWindows()
