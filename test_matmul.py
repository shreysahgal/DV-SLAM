# -*- coding: utf-8 -*-
import numpy as np
import math

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

r = np.array([-0.15640373,  0.00594132, -0.00390095])


# rot = np.array([0.9877899,   0.00292787, -0.1557641,  -0.00390087,  0.9999747,  -0.00594128,
#   0.1557428,   0.00647636,  0.9877764]).reshape(3,3)

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

rot = eulerAnglesToRotationMatrix(r)

# def euler_from_matrix(matrix):
    
# 	# y-x-z Tait–Bryan angles intrincic
# 	# the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    
#     i = 2
#     j = 0
#     k = 1
#     repetition = 0
#     frame = 1
#     parity = 0
	

#     M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
#     if repetition:
#         sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
#         if sy > _EPS:
#             ax = math.atan2( M[i, j],  M[i, k])
#             ay = math.atan2( sy,       M[i, i])
#             az = math.atan2( M[j, i], -M[k, i])
#         else:
#             ax = math.atan2(-M[j, k],  M[j, j])
#             ay = math.atan2( sy,       M[i, i])
#             az = 0.0
#     else:
#         cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
#         if cy > _EPS:
#             ax = math.atan2( M[k, j],  M[k, k])
#             ay = math.atan2(-M[k, i],  cy)
#             az = math.atan2( M[j, i],  M[i, i])
#         else:
#             ax = math.atan2(-M[j, k],  M[j, j])
#             ay = math.atan2(-M[k, i],  cy)
#             az = 0.0

#     if parity:
#         ax, ay, az = -ax, -ay, -az
#     if frame:
#         ax, az = az, ax
#     return ax, ay, az

roteul = rotationMatrixToEulerAngles(rot)


print('original\n', r)
print('rot: \n', rot)
print('back: \n', roteul)

# xzy
# xyz
# yxz
# yzx

# zxy
# zyx