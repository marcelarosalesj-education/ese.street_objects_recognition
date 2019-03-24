"""
Image Difference Compare
"""
import argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='Image Similarity Tool')
parser.add_argument('-i', '--input', nargs=2, required=True,
                    help='Input image')
parser.add_argument('-s', '--scale', nargs='?',
                    help='Scale of original image',
                    default=0.5)
parser.add_argument('-a', '--algorithm', required=True,
                    choices=['abs', 'stif'],
                    help='Algorithm for difference comparison')

def get_resized_image(image, scale=0.5):
    """
    About scaling and resizing:
    https://docs.opencv.org/3.4.2/da/d6e/tutorial_py_geometric_transformations.html
    """
    print("Resizing to {}%".format(scale*100))
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_image

def display_image_information(image):
    """
    Display image shape
    """
    print("Shape: {}x{}".format(image.shape[0], image.shape[1]))


def diff_abs(image1, image2):
    """
    About image differentiation:
    https://stackoverflow.com/questions/51288756/how-do-i-calculate-the-percentage-of-difference-between-two-images-using-python/5128914
    https://stackoverflow.com/questions/843972/image-comparison-fast-algorithm
    """
    diff_image = cv2.absdiff(image1, image2)
    diff_image = diff_image.astype(np.uint8)
    result = (np.count_nonzero(diff_image) * 100)/ diff_image.size
    return result

def get_stif(image):
    """
    get STIF features
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    image = cv2.drawKeypoints(gray, kp, image)
    cv2.imwrite('sift_keypoints_image1.jpg', image)
    return image

def diff_stif(image1, image2):
    """
    shall return the difference percentage based on stif features
    """
    stif_img1 = get_stif(image1)
    stif_img2 = get_stif(image2)
    cv2.imshow("STIF Picture 1", stif_img1)
    cv2.imshow("STIF Picture 2", stif_img2)
    return 0.0

def main():
    """
    main
    """
    # Get input
    args = parser.parse_args()
    filename1 = args.input[0]
    filename2 = args.input[1]
    scale = args.scale
    # Get images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    display_image_information(img1)
    display_image_information(img2)
    # Resize images
    resized_img1 = get_resized_image(img1, scale)
    resized_img2 = get_resized_image(img2, scale)
    cv2.imshow("Resized Picture 1", resized_img1)
    cv2.imshow("Resized Picture 2", resized_img2)
    display_image_information(resized_img1)
    display_image_information(resized_img2)
    # Select difference algorithm
    switcher = {
        'abs': diff_abs,
        'stif': diff_stif,
    }
    result = switcher[args.algorithm](resized_img1, resized_img2)
    print("Result is {}".format(result))
    # Destroy windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
