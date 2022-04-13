import math
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import gtsam
from tqdm import tqdm

import utils


def gtsam_incremental(vertex, edge):

    '''
    Build the graph and do the incremental SLAM


    Arguments:
        vertex {np.ndarray} -- Vertices in format (idx, rotation matrix, x, y, z)
        edge {np.ndarray} -- edges in format (idx0, idx1, rotation matrix, x, y, z)

    Returns:
        result_poses {np.ndarray} -- result poses in format (rotation matrix, x, y, z)
    '''

    isam = gtsam.ISAM2()
    for i in tqdm(range(vertex.shape[0])):
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        idx_p = int(vertex[i, 0])
        rotm = vertex[i, 1:10]
        x_y_z = vertex[i, 10:13]
        T = utils.rp_to_transformation(rotm, x_y_z)
        if idx_p == 0:
            priorNoise = gtsam.noiseModel.Diagonal.Sigmas([5 * math.pi / 180, 5 * math.pi / 180, 5 * math.pi / 180, 0.05, 0.05, 0.05])
            graph.add(gtsam.PriorFactorPose3(idx_p, gtsam.Pose3(T), priorNoise))
            initial.insert(idx_p, gtsam.Pose3(T))
        else:
            prev_pose = result.atPose3(idx_p - 1)
            initial.insert(idx_p, prev_pose)
            for j in range(edge.shape[0]):
                idx1 = int(edge[j, 0])
                idx2 = int(edge[j, 1])
                drotm = edge[j, 2:11]
                dx_y_z = edge[j, 11:14]
                if idx2 == idx_p:
                    dT = utils.rp_to_transformation(drotm, dx_y_z)
                    noise_model = gtsam.noiseModel.Diagonal.Sigmas([5 * math.pi / 180, 5 * math.pi / 180, 5 * math.pi / 180, 0.05, 0.05, 0.05])
                    graph.add(gtsam.BetweenFactorPose3(idx1, idx2, gtsam.Pose3(dT), noise_model));
        isam.update(graph, initial)
        result = isam.calculateEstimate()

    result_poses = gtsam.utilities.extractPose3(result)

    return result_poses


def gtsam_batch(vertex, edge):

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    idx_p = int(vertex[0, 0])
    rotm = vertex[0, 1:10]
    x_y_z = vertex[0, 10:13]
    T = utils.rp_to_transformation(rotm, x_y_z)
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas([5 * math.pi / 180, 5 * math.pi / 180, 5 * math.pi / 180, 0.05, 0.05, 0.05])
    graph.add(gtsam.PriorFactorPose3(idx_p, gtsam.Pose3(T), priorNoise))


    for i in tqdm(range(vertex.shape[0])):
        idx_p = int(vertex[i, 0])
        rotm = vertex[i, 1:10]
        x_y_z = vertex[i, 10:13]
        T = utils.rp_to_transformation(rotm, x_y_z)
        initial.insert(idx_p, gtsam.Pose3(T))

    for j in tqdm(range(edge.shape[0])):
        idx1 = int(edge[j, 0])
        idx2 = int(edge[j, 1])
        drotm = edge[j, 2:11]
        dx_y_z = edge[j, 11:14]
        dT = utils.rp_to_transformation(drotm, dx_y_z)
        noise_model = gtsam.noiseModel.Diagonal.Sigmas([5 * math.pi / 180, 5 * math.pi / 180, 5 * math.pi / 180, 0.05, 0.05, 0.05])
        graph.add(gtsam.BetweenFactorPose3(idx1, idx2, gtsam.Pose3(dT), noise_model));

    optimizer = gtsam.GaussNewtonOptimizer(graph, initial);
    result = optimizer.optimize();
    result_poses = gtsam.utilities.extractPose3(result)

    return result_poses

def main():
    data = np.loadtxt("out_09.txt", delimiter=",", unpack=False)
    vertex = utils.data_to_vertex(data)
    edge = utils.vertex_to_edge(vertex)

    closure_edge = np.loadtxt("closure_edge_09.txt", unpack=False)
    edge = np.vstack((edge, closure_edge))

    print("Calculating result")
    result_poses = gtsam_incremental(vertex, edge)

    np.savetxt('gtsam_result_poses_09.txt', result_poses)

    ground_truth = np.loadtxt("gt_pose_09.txt", unpack=False)
    plt.plot(vertex[:, 10], vertex[:, 12], 'blue', label='DeepVO')
    plt.plot(result_poses[:, 9], result_poses[:, 11], 'red', label='Gtsam result')
    plt.plot(ground_truth[:, 3], ground_truth[:, 11], 'green', label='Ground truth')
    plt.legend(['DeepVO', 'Gtsam result', 'Ground Truth'])

    plt.savefig('gtsam_09.png')
    plt.show()


if __name__ == '__main__':
    main()
