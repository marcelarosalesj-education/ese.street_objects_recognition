"""
Use similarity tool to compare all images in street_objects_recognition/images
"""

import os
import pdb
from helpers import Komander

# Get Images
img_dir = '../images/'
images = os.listdir(img_dir)
images = [img_dir+img for img in images if '.jpg' in img]

# Similarity Tool Options
options = ['abs', 'sift', 'kaze', 'surf', 'ssim']

# Results file
results_file = 'results_compare_all.txt'

for opt in options:
    for img1 in images:
        for img2 in images:
            if img1 != img2:
                # Execute command
                command = 'python3 main.py -i {} {} -a {} '.format(img1, img2, opt)
                res = Komander.run(command)
                # Get the result
                result = str(res.stdout)
                result = result[ result.find(':')+1: result.find('%')  ]
                result = result.replace(' ','')
                # Add to results file
                with open(results_file, 'a') as filetowrite:
                    filetowrite.write('{}\t{}\t{}\t{}\n'.format(opt, result, img1, img2))

