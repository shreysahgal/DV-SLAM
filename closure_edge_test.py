import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

import mono_vo

if __name__ == '__main__':

    select_closures = np.loadtxt("select_closures_07.txt", unpack=False)
    closure_edge = np.zeros((len(select_closures), 14))

    img_path = '/home/shanzhaowang99/Desktop/EECS568/Project/deepvo_gtsam/kitti/07/image_2'
    filenames = [img for img in glob.glob(img_path + "/*.png")]
    filenames.sort()

    images = []
    for fname in tqdm(filenames):
        img = cv.imread(fname)
        images.append(img)

    closure_vo = mono_vo.MonoOdometery(img_path)

    for idx in tqdm(range(select_closures.shape[0])):
        i, j = select_closures[idx]
        i = int(i)
        j = int(j)
        closure_vo.img_to_frame(images[i], images[j])
        R, t = closure_vo.visual_odometery()
        # print('pair image %d %d' % (i, j))
        # print(R)
        # print(t)
        closure_edge[idx, 0] = i
        closure_edge[idx, 1] = j
        closure_edge[idx, 2:5] = R[0]
        closure_edge[idx, 5:8] = R[1]
        closure_edge[idx, 8:11] = R[2]
        closure_edge[idx, 11:14] = t.flatten()

    np.savetxt('closure_edge_07.txt', closure_edge)