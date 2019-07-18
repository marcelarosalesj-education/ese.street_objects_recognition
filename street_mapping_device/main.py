"""
Street mapping device - Proof of concept
"""

import cv2
import argparse
import os
import skimage.measure
import imutils
import matlab.engine
import sys
import shutil

TIME_FRAME = 10
LONG_LINE = 60
SIMILARITY_ACCEPTANCE = 70


results_directory = ''
parser = argparse.ArgumentParser(description='Street mapping device - Proof of concept')
parser.add_argument('-i', '--input', required=True,
                    help='Video input')
parser.add_argument('-o', '--output', required=False,
                    default='Results',
                    help='Results directory')
parser.add_argument('-sa', '--similarity_algorithm', required=False,
                    choices=['sift', 'kaze', 'surf', 'ssim'],
                    default='ssim',
                    help='Algorithm for difference comparison')

def add_metadata(filename, info):
    os.system('exiftool {} -overwrite_original -q -description="{}"'.format(filename, info))


def object_recognition(image):
    eng = matlab.engine.start_matlab()
    var, per = eng.use_nn(image, nargout=2)
    return var

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

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    except cv2.error:
        print('There was an error in SIFT Match phase')
        return -1

    good_points = []
    ratio = 0.6
    for m, n in matches:
        # The less the distance, the better the matches
        if m.distance < ratio*n.distance:
            good_points.append(m)

    # How similar they are
    number_keypoints = 0
    number_keypoints = (
        len(key_points_1) if len(key_points_1) >= len(key_points_2)
        else len(key_points_2))

    result = cv2.drawMatches(image1, key_points_1, image2, key_points_2, good_points, None)
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
    kaze = cv2.KAZE_create()
    key_points_1, desc_1 = kaze.detectAndCompute(image1, None)
    key_points_2, desc_2 = kaze.detectAndCompute(image2, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    except cv2.error:
        print('There was an error in KAZE Match phase')
        return -1

    good_points = []
    ratio = 0.6
    for m, n in matches:
        # The less the distance, the better the matches
        if m.distance < ratio*n.distance:
            good_points.append(m)

    # How similar they are
    number_keypoints = 0
    number_keypoints = (
        len(key_points_1) if len(key_points_1) >= len(key_points_2)
        else len(key_points_2))

    result = cv2.drawMatches(image1, key_points_1, image2, key_points_2, good_points, None)
    return len(good_points) / number_keypoints * 100

def diff_surf(image1, image2):
    """Calculates how similar are two images based on SURF descriptors


    Args:
        param1: image
        param2: image
    Returns:
        float: percentage

    """
    surf = cv2.xfeatures2d.SURF_create()
    key_points_1, desc_1 = surf.detectAndCompute(image1, None)
    key_points_2, desc_2 = surf.detectAndCompute(image2, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc_1, desc_2, k=2)
    except cv2.error:
        print('There was an error in SURF Match phase')
        return -1

    good_points = []
    ratio = 0.6
    for m, n in matches:
        # The less the distance, the better the matches
        if m.distance < ratio*n.distance:
            good_points.append(m)

    # How similar they are
    number_keypoints = 0
    number_keypoints = (
        len(key_points_1) if len(key_points_1) >= len(key_points_2)
        else len(key_points_2))

    result = cv2.drawMatches(image1, key_points_1,
                             image2, key_points_2,
                             good_points, None)
    return len(good_points) / number_keypoints * 100

def diff_ssim(image1, image2):
    """ Calculates how similar are two images based on SSIM algorithm
    Args:
        param1: image
        param2: image
    Returns:
        float: percentage

    Note that SSIM score goes from -1 to 1
    https://github.com/jterrace/pyssim/issues/15
    """
    (score, diff) = skimage.measure.compare_ssim(image1, image2, full=True)
    # Scale score from [-1,1] to [0,1], then convert to percentage
    score = (score+1)/2
    score = score * 100
    # Scale diff differences matrix [-1,1] to [0,255]
    diff = (diff * 255).astype('uint8')

    # Apply a threshold to the differences diff image
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return score

def compare_with_last(image, last_image, similarity_algorithm):
    if similarity_algorithm == 'ssim':
        score = diff_ssim(image, last_image)
    elif similarity_algorithm == 'sift':
        score = diff_sift(image, last_image)
    elif similarity_algorithm == 'kaze':
        score = diff_kaze(image, last_image)
    elif similarity_algorithm == 'surf':
        score = diff_surf(image, last_image)
    else:
        print('Invalid similarity algorithm')
        sys.exit()
    print('Score: {}'.format(score))
    if score < SIMILARITY_ACCEPTANCE:
        return True 
    else:
        return False


def main():
    args = parser.parse_args()
    video_file = args.input
    similarity_algorithm = args.similarity_algorithm
    vidcap = cv2.VideoCapture(video_file)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    results_directory = args.output

    try:
        os.mkdir(results_directory)
    except FileExistsError as e:
        print('W: {}'.format(e))
        print('W: Directory was removed.')
        shutil.rmtree(results_directory)
        os.mkdir(results_directory)

    success, image = vidcap.read()
    count = 0
    image_number = 0
    frames_analyzed = 0
    while success:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not count % TIME_FRAME:
            frames_analyzed += 1
            if image_number == 0:
                keep = True
            else:
                last_image_path = '{}/frame{:02d}.jpg'.format(results_directory, image_number - 1)
                last_image = cv2.imread(last_image_path)
                last_image_gray = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
                keep = compare_with_last(image_gray, last_image_gray, similarity_algorithm)
            if keep == True:
                new_file = '{}/frame{:02d}.jpg'.format(results_directory, image_number)
                cv2.imwrite(new_file, image)
                meta = object_recognition(new_file)
                add_metadata(new_file, meta)
                image_number += 1
        count += 1
        success, image = vidcap.read()
    print('-'*LONG_LINE)
    print('Street mapping device analysis')
    print('- Video: {}'.format(video_file))
    print('- Similarity algorithm: {}'.format(similarity_algorithm))
    print('- Frames in the video {}'.format(length))
    print('- 1 frame every {} frames'.format(TIME_FRAME))
    print('- Frames analyzed:  {}'.format(frames_analyzed))
    print('- Results in: {}'.format(results_directory))
    print('- Number of images stored: {}'.format(len(os.listdir(results_directory))))
    print('-'*LONG_LINE)

if __name__ == '__main__':
    main()
