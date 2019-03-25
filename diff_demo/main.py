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
                    choices=['abs', 'sift'],
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
    """ Calculates how different are two images based on Absolute difference

    Args:
        param1: image
        param2: image
    Returns:
        float: percentage

    """
    diff_image = cv2.absdiff(image1, image2)
    diff_image = diff_image.astype(np.uint8)
    result = (np.count_nonzero(diff_image) * 100)/ diff_image.size
    return result

def get_sift(image):
    """
    get sift features
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points = sift.detectAndCompute(gray, None)
    image = cv2.drawKeypoints(gray, key_points, image)
    return image

def diff_sift(image1, image2):
    """Calculates how similar are two images based on SIFT features

    https://pysource.com/2018/07/20/detect-how-similar-two-images-are-with-opencv-and-python/

    Two images are similar if they have many good points in common, but as the size of the image is
    variant we need to calculate proportionally.

    Args:
        param1: image
        param2: image
    Returns:
        float: percentage

    """
    sift = cv2.xfeatures2d.SIFT_create()
    key_points_1, desc_1 = sift.detectAndCompute(image1, None)
    key_points_2, desc_2 = sift.detectAndCompute(image2, None)
    print('Key Points 1st: {}'.format(len(key_points_1)))
    print('Key Points 2nd: {}'.format(len(key_points_2)))

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    ratio = 0.6
    for m, n in matches:
        # The less the distance, the better the matches
        if m.distance < ratio*n.distance:
            good_points.append(m)
    print('Good Points: {}'.format(len(good_points)))

    # How similar they are
    number_keypoints = 0
    number_keypoints = (
        len(key_points_1) if len(key_points_1) >= len(key_points_2)
        else len(key_points_2))

    result = cv2.drawMatches(image1, key_points_1, image2, key_points_2, good_points, None)
    cv2.imshow('result', result)
    return len(good_points) / number_keypoints * 100

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
        'sift': diff_sift,
    }
    result = switcher[args.algorithm](resized_img1, resized_img2)
    print("Result is {0:.2f} %".format(result))
    # Destroy windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
