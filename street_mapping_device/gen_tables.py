"""
Generate tables
"""

import cv2
import argparse
import os
import skimage.measure
import imutils
import matlab.engine
import sys
import numpy as np

TIME_FRAME = 10
SIZE_W = 0
SIZE_H = 0
NUM_W = 5
LONG_LINE = 60
results_directory = ''
parser = argparse.ArgumentParser(description='Street mapping device - Proof of concept')
parser.add_argument('-v', '--video', required=True,
                    help='Video input')
parser.add_argument('-r', '--results', required=False,
                    default='Results',
                    help='Results directory')

def main():
    args = parser.parse_args()
    video_file = args.video
    vidcap = cv2.VideoCapture(video_file)
    frames_total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    results_directory = args.results
    # SIZE 
    success, image = vidcap.read()
    SIZE_H = image.shape[0] // 10
    SIZE_W = image.shape[1] // 10
    # Gen table for 1 frame per TIME_FRAME
    store_frames = frames_total // TIME_FRAME
    height = store_frames // NUM_W
    if store_frames % NUM_W:
        height = height + 1
    height = height * SIZE_H
    width = NUM_W * SIZE_W
    chan = 3
    video_table = np.zeros((height, width, chan), np.uint8)
    # Counting values 
    x_offset = 0
    y_offset = 0
    count = 0
    frames_analyzed = 0
    while success:
        image = cv2.resize(image, (SIZE_W, SIZE_H))
        if not count % TIME_FRAME:
            frames_analyzed += 1
            try:
                video_table[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
            except ValueError as e:
                print(e)
                cv2.imwrite('results-incomplete.jpg', video_table)
                import ipdb; ipdb.set_trace()
                
            x_offset = x_offset + SIZE_W
            if x_offset >= (SIZE_W * NUM_W):
                y_offset = y_offset + SIZE_H
                x_offset = 0
        count += 1
        success, image = vidcap.read()

    cv2.imwrite('results-complete.jpg', video_table)
    print('-'*LONG_LINE)
    print('- Video: {}'.format(video_file))
    print('- Frames in the video {}'.format(frames_total))
    print('- 1 frame every {} frames'.format(TIME_FRAME))
    print('- Frames analyzed:  {}'.format(frames_analyzed))
    print('-'*LONG_LINE)

if __name__ == '__main__':
    main()
