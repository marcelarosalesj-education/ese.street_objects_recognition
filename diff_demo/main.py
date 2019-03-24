"""
Image Difference Compare
"""
import sys
import cv2
import numpy as np

def get_resized_image(image, scale = 0.5):
    """
    About scaling and resizing:
    https://docs.opencv.org/3.4.2/da/d6e/tutorial_py_geometric_transformations.html
    """
    print("Resizing to {}%".format(scale*100))
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_image

def display_image_information(image):
    print("Shape: {}x{}".format(image.shape[0], image.shape[1]))


def diff_algorithm_abs(image1, image2):
    """
    About image differentiation:
    https://stackoverflow.com/questions/51288756/how-do-i-calculate-the-percentage-of-difference-between-two-images-using-python/5128914
    https://stackoverflow.com/questions/843972/image-comparison-fast-algorithm
    """
    diff_image = cv2.absdiff(image1, image2)
    diff_image = diff_image.astype(np.uint8)
    result = (np.count_nonzero(diff_image) * 100)/ diff_image.size
    return result


filename1 = sys.argv[1]
filename2 = sys.argv[2]
scale = float(sys.argv[3])

img1 = cv2.imread(filename1, 0)
img2 = cv2.imread(filename2, 0)

display_image_information(img1)
display_image_information(img2)

resized_img1 = get_resized_image(img1, scale)
resized_img2 = get_resized_image(img2, scale)

cv2.imshow("Resized Picture 1", resized_img1)
cv2.imshow("Resized Picture 2", resized_img2)

display_image_information(resized_img1)
display_image_information(resized_img2)

result = diff_algorithm_abs(resized_img1, resized_img2)
print("Result is {}".format(result))

cv2.waitKey(0)
cv2.destroyAllWindows()
