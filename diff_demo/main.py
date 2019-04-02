"""
Image Difference Compare
"""
import argparse
import cv2
import imutils
import numpy as np
import skimage.measure
import logging as log

DISPLAY = False
VERBOSITY = 0

parser = argparse.ArgumentParser(description='Image Similarity Tool')
parser.add_argument('-i', '--input', nargs=2, required=True,
                    help='Two input images')
parser.add_argument('-s', '--scale', nargs='?',
                    help='Scale of original image',
                    default=0.5)
parser.add_argument('-a', '--algorithm', required=True,
                    choices=['abs', 'sift', 'kaze', 'surf', 'ssim'],
                    help='Algorithm for difference comparison')
parser.add_argument('-c', '--colorspace', default='bgr',
                    choices=['bgr', 'gray'],
                    help='Select colorspace for processing images')
parser.add_argument('-d', '--display',
                    help='Display result images',
                    action='store_true',
                    default=False)
parser.add_argument('-v', '--verbose',
                    help='Increase output verbosity',
                    action='count',
                    default=0)

def get_resized_image(image, scale=0.5):
    """
    About scaling and resizing:
    https://docs.opencv.org/3.4.2/da/d6e/tutorial_py_geometric_transformations.html
    """
    log.info("Resizing to {}%".format(scale*100))
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_image

def display_image_information(image, header=''):
    """
    Display image shape
    """
    log.info('********************')
    if header:
        log.info(header)
    log.info('Shape: {}'.format(image.shape))
    log.info('********************')


def generate_random_complementary_images():
    """ Generate random complementary images and save the result
    """
    from PIL import Image
    random_img = np.random.randint(0, 255, (1000, 1000)).astype(np.uint8)
    result = Image.fromarray(random_img)
    result.save('out_orig.bmp')
    result = Image.fromarray(255-random_img)
    result.save('out_comp.bmp')

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
    global DISPLAY
    sift = cv2.xfeatures2d.SIFT_create()
    key_points_1, desc_1 = sift.detectAndCompute(image1, None)
    key_points_2, desc_2 = sift.detectAndCompute(image2, None)
    log.info('Key Points 1st: {}'.format(len(key_points_1)))
    log.info('Key Points 2nd: {}'.format(len(key_points_2)))

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    except cv2.error:
        log.error('There was an error in SIFT Match phase')
        return -1

    good_points = []
    ratio = 0.6
    for m, n in matches:
        # The less the distance, the better the matches
        if m.distance < ratio*n.distance:
            good_points.append(m)
    log.info('Good Points: {}'.format(len(good_points)))

    # How similar they are
    number_keypoints = 0
    number_keypoints = (
        len(key_points_1) if len(key_points_1) >= len(key_points_2)
        else len(key_points_2))

    result = cv2.drawMatches(image1, key_points_1, image2, key_points_2, good_points, None)
    if DISPLAY:
        cv2.imshow('result_sift', result)
    return len(good_points) / number_keypoints * 100

def diff_kaze(image1, image2):
    """Calculates how similar are two images based on KAZE descriptors

    https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774

    Args:
        param1: image
        param2: image
    Returns:
        float: percentage

    """
    global DISPLAY
    kaze = cv2.KAZE_create()
    key_points_1, desc_1 = kaze.detectAndCompute(image1, None)
    key_points_2, desc_2 = kaze.detectAndCompute(image2, None)
    log.info('Key Points 1st: {}'.format(len(key_points_1)))
    log.info('Key Points 2nd: {}'.format(len(key_points_2)))

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    except cv2.error:
        log.error('There was an error in KAZE Match phase')
        return -1

    good_points = []
    ratio = 0.6
    for m, n in matches:
        # The less the distance, the better the matches
        if m.distance < ratio*n.distance:
            good_points.append(m)
    log.info('Good Points: {}'.format(len(good_points)))

    # How similar they are
    number_keypoints = 0
    number_keypoints = (
        len(key_points_1) if len(key_points_1) >= len(key_points_2)
        else len(key_points_2))

    result = cv2.drawMatches(image1, key_points_1, image2, key_points_2, good_points, None)
    if DISPLAY:
        cv2.imshow('result_kaze', result)
    return len(good_points) / number_keypoints * 100

def diff_surf(image1, image2):
    """Calculates how similar are two images based on SURF descriptors


    Args:
        param1: image
        param2: image
    Returns:
        float: percentage

    """
    global DISPLAY
    surf = cv2.xfeatures2d.SURF_create()
    key_points_1, desc_1 = surf.detectAndCompute(image1, None)
    key_points_2, desc_2 = surf.detectAndCompute(image2, None)
    log.info('Key Points 1st: {}'.format(len(key_points_1)))
    log.info('Key Points 2nd: {}'.format(len(key_points_2)))

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    except cv2.error:
        log.error('There was an error in SURF Match phase')
        return -1

    good_points = []
    ratio = 0.6
    for m, n in matches:
        # The less the distance, the better the matches
        if m.distance < ratio*n.distance:
            good_points.append(m)
    log.info('Good Points: {}'.format(len(good_points)))

    # How similar they are
    number_keypoints = 0
    number_keypoints = (
        len(key_points_1) if len(key_points_1) >= len(key_points_2)
        else len(key_points_2))

    result = cv2.drawMatches(image1, key_points_1,
                             image2, key_points_2,
                             good_points, None)
    if DISPLAY:
        cv2.imshow('result_surf', result)
    return len(good_points) / number_keypoints * 100

def diff_ssim(image1, image2):
    """ Calculates how similar are two images based on SSIM algorithm

    https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim

    Args:
        param1: image
        param2: image
    Returns:
        float: percentage

    Note that SSIM score goes from -1 to 1
    https://github.com/jterrace/pyssim/issues/15
    """
    global DISPLAY
    (score, diff) = skimage.measure.compare_ssim(image1, image2, full=True)
    # Scale score from [-1,1] to [0,1], then convert to percentage
    score = (score+1)/2
    score = score * 100
    # Scale diff differences matrix [-1,1] to [0,255]
    diff = (diff * 255).astype('uint8')

    # Apply a threshold to the differences diff image
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Draw the contours on the original images
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Show the output images
    if DISPLAY:
        cv2.imshow("Original", image1)
        cv2.imshow("Modified", image2)
        cv2.imshow("Diff", diff)
        cv2.imshow("Thresh", thresh)
    return score


def main():
    """
    main
    """
    # Parse arguments
    args = parser.parse_args()
    # Get input
    filename1 = args.input[0]
    filename2 = args.input[1]
    scale = args.scale
    # Display options
    global DISPLAY
    if args.display:
        DISPLAY = True
    # Set up verbosity options
    if args.verbose == 0:
        log.basicConfig(format='%(levelname)s: %(message)s')
    else:
        log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG)
        log.info('Verbose output')
    # Get images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    # Use selected colorspace
    if args.colorspace == 'gray' or args.algorithm == 'ssim':
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Verify shape
    if img1.shape != img2.shape:
        log.error('Images have different shapes, cannot perform a comparison.')
        return -1
    display_image_information(img1, header='Original Image 1')
    display_image_information(img2, header='Original Image 2')
    # Resize images
    resized_img1 = get_resized_image(img1, scale)
    resized_img2 = get_resized_image(img2, scale)
    if DISPLAY:
        cv2.imshow("Resized Picture 1", resized_img1)
        cv2.imshow("Resized Picture 2", resized_img2)
    display_image_information(resized_img1, header='Resized Image 1')
    display_image_information(resized_img2, header='Resized Image 2')
    # Select difference algorithm
    switcher = {
        'abs': diff_abs,
        'sift': diff_sift,
        'kaze': diff_kaze,
        'surf': diff_surf,
        'ssim': diff_ssim,
    }
    result = switcher[args.algorithm](resized_img1, resized_img2)
    if result >= 0:
        print("Similarity result: {0:.2f} %".format(result))
    # Destroy windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
