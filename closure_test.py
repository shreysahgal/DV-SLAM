import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

import mono_vo
import loop_closure

if __name__ == '__main__':


    print('Reading images...')
    img_path = '/home/shanzhaowang99/Desktop/EECS568/Project/deepvo_gtsam/kitti/07/image_2'
    filenames = [img for img in glob.glob(img_path + "/*.png")]
    filenames.sort()

    images = []
    for fname in tqdm(filenames):
        img = cv.imread(fname)
        images.append(img)

    print('Traing cluster algorithm...')
    dictionary = loop_closure.makeSiftVocab(images, train_size=100, dict_size=200, retries=5)

    print('BoW-vectorizing images...')
    bowvecs = loop_closure.bowVectorizeImages(images, dictionary)

    print('Detecting closures...')
    closures = loop_closure.detectClosures(bowvecs,thold=0.10)

    print('Done! Found %d closures.' % len(closures))

    # print(closures)

    gap = 800
    select_closures = []
    for i, j in closures:
        if j > i + gap :
            select_closures.append((i, j))

    print('Found %d select_closures.' % len(select_closures))

    with open('select_closures_07.txt', 'w') as f:
        for select_closures_idx in select_closures:
            f.write('%d %d\n' % select_closures_idx)