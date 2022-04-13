import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

import loop_closure
import edge_calculation
import utils
import gtsam_solver


if __name__ == '__main__':

    '''
    Fill in:
    img_path - path of the Kitti dataset images e.g. ./kitti/07/image_2
    vo_data_path  - path of the output of DeepVO e.g. ./DeepVO/result/out_07.txt
    gt_path - path of the Kitti ground truth poses e.g. ./kitti/poses/07.txt
    
    '''

    img_path = './kitti/07/image_2'
    vo_data_path = './DeepVO/result/out_07.txt'
    gt_path = './kitti/poses/07.txt'


    print('Reading images...')
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

    # Select closures
    gap = 800
    select_closures = []
    for i, j in closures:
        if j > i + gap :
            select_closures.append((i, j))

    print('Found %d select_closures.' % len(select_closures))

    # Calculate edges between closure vertices
    closure_edge = np.zeros((len(select_closures), 14))
    closure_vo = edge_calculation.MonoOdometery(img_path)


    print('Calculating edges...')
    for idx in tqdm(range(len(select_closures))):
        i, j = select_closures[idx]
        i = int(i)
        j = int(j)
        closure_vo.img_to_frame(images[i], images[j])
        R, t = closure_vo.visual_odometery()
        closure_edge[idx, 0] = i
        closure_edge[idx, 1] = j
        closure_edge[idx, 2:5] = R[0]
        closure_edge[idx, 5:8] = R[1]
        closure_edge[idx, 8:11] = R[2]
        closure_edge[idx, 11:14] = t.flatten()


    # Solve the problem using gtsam solver
    vo_data = np.loadtxt(vo_data_path, delimiter=",", unpack=False)
    vertex = utils.data_to_vertex(vo_data)
    edge = utils.vertex_to_edge(vertex)
    edge = np.vstack((edge, closure_edge))

    print("Calculating result")
    result_poses = gtsam_solver.gtsam_incremental(vertex, edge)

    # Save the results in format (rows of rotation matrix, x, y, z)
    np.savetxt('DV_SLAM_result_poses.txt', result_poses)

    ground_truth = np.loadtxt(gt_path, unpack=False)
    plt.plot(vertex[:, 10], vertex[:, 12], 'blue', label='DeepVO')
    plt.plot(result_poses[:, 9], result_poses[:, 11], 'red', label='Gtsam result')
    plt.plot(ground_truth[:, 3], ground_truth[:, 11], 'green', label='Ground truth')
    plt.legend(['DeepVO', 'DV SLAM result', 'Ground Truth'])

    plt.savefig('DV_SLAM_result.png')
    plt.show()