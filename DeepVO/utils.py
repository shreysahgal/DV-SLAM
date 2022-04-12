import math
import numpy as np


def eul2rotm(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def eul2rotm_y_x_z(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[1]), -math.sin(theta[1]) ],
                    [0,         math.sin(theta[1]), math.cos(theta[1])  ]
                    ])

    R_y = np.array([[math.cos(theta[0]),    0,      math.sin(theta[0])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[0]),   0,      math.cos(theta[0])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_y, np.dot( R_x, R_z ))

    return R

# Get 4x4 transformation matrix
def rp_to_transformation(rotm, P):
    R = np.vstack((rotm[0:3], rotm[3:6], rotm[6:9]))
    P = np.reshape(P, (3,1))
    T = np.vstack((np.hstack((R, P)), np.array([0, 0, 0, 1])))
    return T


# data in format (theta y, theta x, theta z, x, y, z)
# vertex in format (idx, rotation matrix, x, y, z)
def data_to_vertex(data):
    vertex = np.zeros((data.shape[0], 13))
    for i in range(vertex.shape[0]):
        vertex[i,0] = i;
        R = eul2rotm_y_x_z(data[i, 1:4])
        vertex[i, 1:4] = R[0]
        vertex[i, 4:7] = R[1]
        vertex[i, 7:10] = R[2]
        vertex[i, 10:13] = data[i, 3:6]

    return vertex

# edge in format (idx0, idx1, rotation matrix, x, y, z)
def vertex_to_edge(vertex):
    edge = np.zeros((vertex.shape[0] - 1, 14))
    for i in range(edge.shape[0]):
        pose_0 = rp_to_transformation(vertex[i, 1:10], vertex[i, 10:13])
        pose_1 = rp_to_transformation(vertex[i+1, 1:10], vertex[i+1, 10:13])
        T = np.dot(np.linalg.inv(pose_0), pose_1)
        edge[i, 0] = i
        edge[i, 1] = i+1
        edge[i, 2:11] = np.hstack((T[0, :3], T[1, :3], T[2, :3]))
        edge[i, 11:14] = T[0:3, 3]

    return edge



