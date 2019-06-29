"""
Street mapping device - Proof of concept
"""

import cv2
import argparse
import os
import skimage.measure
import imutils
import matlab.engine


TIME_FRAME = 10
LONG_LINE = 60
SIMILARITY_ACCEPTANCE = 70

results_directory = 'Results'
parser = argparse.ArgumentParser(description='Street mapping device - Proof of concept')
parser.add_argument('-i', '--input', required=True,
                    help='Video input')
parser.add_argument('-sa', '--similarity_algorithm', required=False,
                    choices=['sift', 'kaze', 'surf', 'ssim'],
                    default='ssim',
                    help='Algorithm for difference comparison')

def add_metadata(filename, info):
    os.system('exiftool {} -overwrite_original -description="street object map identified is {}"'.format(filename, info))


def object_recognition(image):
    eng = matlab.engine.start_matlab()
    var = eng.use_nn(image)
    return var[0][0]

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

def compare_with_last(image, last_image):
    score = diff_ssim(image, last_image)
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

    print('-'*LONG_LINE)
    print('Street mapping device analysis')
    print('- Video: {}'.format(video_file))
    print('- Frames in the video {}'.format(length))
    print('- 1 frame every {} frames'.format(TIME_FRAME))
    print('- Similarity algorithm: {}'.format(similarity_algorithm))
    print('-'*LONG_LINE)

    try:
        os.mkdir(results_directory)
    except FileExistsError as e:
        print('W: {}'.format(e))

    success, image = vidcap.read()
    count = 0
    image_number = 0
    while success:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not count % TIME_FRAME:
            if image_number == 0:
                keep = True
            else:
                last_image_path = '{}/frame{}.jpg'.format(results_directory, image_number - 1)
                last_image = cv2.imread(last_image_path)
                last_image_gray = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
                keep = compare_with_last(image_gray, last_image_gray)
            if keep == True:
                new_file = '{}/frame{}.jpg'.format(results_directory, image_number)
                cv2.imwrite(new_file, image)
                meta = object_recognition(new_file)
                add_metadata(new_file, meta)
                image_number += 1
        count += 1
        success, image = vidcap.read()

if __name__ == '__main__':
    main()
