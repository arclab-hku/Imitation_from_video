""" This file defines a mesh as a tuple of (vertices, triangles)
All operations are based on numpy ndarray
- vertices: np ndarray of shape (n, 3) np.float32
- triangles: np ndarray of shape (n_, 3) np.uint32
"""
import numpy as np

def box_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
        rpy= np.zeros(3), # euler angle (in rad) not implemented yet.
    ):
    if not (rpy == 0).all():
        raise NotImplementedError("Only axis-aligned box triangle mesh is implemented")

    vertices = np.empty((8, 3), dtype= np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2
    print("***** V ", vertices.shape)

    triangles = -np.ones((12, 3), dtype= np.uint32)
    triangles[0] = [0, 2, 1] #
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2] #
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6] #
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5] #
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4] #
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3] #
    triangles[11]= [3, 6, 7]
    print("***** T ", triangles.shape, triangles)

    return vertices, triangles


def frame_trimesh(
        size,            # float [3] for x, y, z axis length (in meter) under box frame
        inner_frame_scale, # how much percentage of the inner frame account for the outter frame
        center_position, # float [3] position (in meter) in world frame
        rpy= np.zeros(3), # euler angle (in rad) not implemented yet.
    ):
    if not (rpy == 0).all():
        raise NotImplementedError("Only axis-aligned box triangle mesh is implemented")

    vertices = np.empty((16, 3), dtype= np.float32)
    vertices[:] = center_position
    # outer
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2

    # inner
    vertices[8, 0] -= size[0] / 2
    vertices[8, 1] -= size[1] / 2 * inner_frame_scale
    vertices[8, 2] += size[1] / 2 * inner_frame_scale
    vertices[9, 0] += size[0] / 2
    vertices[9, 1] -= size[1] / 2 * inner_frame_scale
    vertices[9, 2] += size[1] / 2 * inner_frame_scale
    vertices[10, 0] -= size[0] / 2
    vertices[10, 1] -= size[1] / 2 * inner_frame_scale
    vertices[10, 2] -= size[1] / 2 * inner_frame_scale
    vertices[11, 0] += size[0] / 2
    vertices[11, 1] -= size[1] / 2 * inner_frame_scale
    vertices[11, 2] -= size[1] / 2 * inner_frame_scale
    vertices[12, 0] -= size[0] / 2
    vertices[12, 1] += size[1] / 2 * inner_frame_scale
    vertices[12, 2] += size[1] / 2 * inner_frame_scale
    vertices[13, 0] += size[0] / 2
    vertices[13, 1] += size[1] / 2 * inner_frame_scale
    vertices[13, 2] += size[1] / 2 * inner_frame_scale
    vertices[14, 0] -= size[0] / 2
    vertices[14, 1] += size[1] / 2 * inner_frame_scale
    vertices[14, 2] -= size[1] / 2 * inner_frame_scale
    vertices[15, 0] += size[0] / 2
    vertices[15, 1] += size[1] / 2 * inner_frame_scale
    vertices[15, 2] -= size[1] / 2 * inner_frame_scale

    triangles = -np.ones((32, 3), dtype= np.uint32)
    triangles[0] = [0, 8, 2]  #
    triangles[1] = [2, 8, 10]  #
    triangles[2] = [0, 4, 8]  #
    triangles[3] = [8, 4, 12]  #
    triangles[4] = [12, 4, 6]  #
    triangles[5] = [12, 6, 14]  #
    triangles[6] = [10, 14, 6]  #
    triangles[7] = [10, 6, 2]  #

    triangles[8] = [9, 1, 3]  #
    triangles[9] = [9, 3, 11]  #
    triangles[10] = [11, 3, 7]  #
    triangles[11] = [11, 7, 15]  #
    triangles[12] = [15, 7, 13]  #
    triangles[13] = [13, 7, 5]  #
    triangles[14] = [1, 13, 5]  #
    triangles[15] = [1, 9, 13]  #

    triangles[16] = [0, 2, 1] #
    triangles[17] = [1, 2, 3]
    triangles[18] = [4, 5, 6] #
    triangles[19] = [5, 7, 6]
    triangles[20] = [0, 1, 4] #
    triangles[21] = [1, 5, 4]
    triangles[22] = [2, 6, 3] #
    triangles[23] = [3, 6, 7]
    triangles[24] = [8, 9, 10]
    triangles[25] = [10, 9, 11]
    triangles[26] = [8, 13, 9]
    triangles[27] = [12, 13, 8]
    triangles[28] = [14, 15, 13]
    triangles[29] = [14, 13, 12]
    triangles[30] = [10, 11, 15]
    triangles[31] = [10, 15, 14]
    # triangles.astype(np.int32, copy=False)
    return vertices, triangles

def combine_trimeshes(*trimeshes):
    if len(trimeshes) > 2:
        return combine_trimeshes(
            trimeshes[0],
            combine_trimeshes(*trimeshes[1:])
        )
    trimesh_0, trimesh_1 = trimeshes

    if trimesh_0[1].shape[0] < trimesh_1[1].shape[0]:
        trimesh_0, trimesh_1 = trimesh_1, trimesh_0
    
    trimesh_1 = (trimesh_1[0], trimesh_1[1] + trimesh_0[0].shape[0])
    vertices = np.concatenate((trimesh_0[0], trimesh_1[0]), axis= 0)
    triangles = np.concatenate((trimesh_0[1], trimesh_1[1]), axis= 0)

    return vertices, triangles

def move_trimesh(trimesh, move: np.ndarray):
    """ inplace operation """
    trimesh[0] += move
