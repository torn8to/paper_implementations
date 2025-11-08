# distutils: language = c++
# cython: language_level=3
import numpy as np 
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

from cython_modules.voxel_types cimport Voxel2d, VoxelMap2d

cpdef np.ndarray unique_voxel_map_image(np.ndarray[np.int32_t, ndim = 2] cloud_xy_voxels,
                                       float max_range=100.0,
                                       float resolution=0.5):
    ''' Performs unique operation in numpy with count and draws 
        them into an image at the core it works similarly to an MFU 

    Parameters
    __________

    cloud_xy_voxels: np.ndarray[np.int32_t, ndim = 2]
       this is N, 2 list of xy coordinates voxelized using a resolution factor

    max_range: float 
        the max range considered when drawing the image assuming
        the point cloud is already clipped to this max_range
     
    resolution: float
       this is the xy voxel dimension size it is assumed to be a square
       the point cloud is already clipped to this max_range

    Returns
    _______
        returns a voxelized image count where the counted occurrence of voxels is converted to 
        an image representation
    '''
    
    cdef int num_points = cloud_xy_voxels.shape[0]
    cdef VoxelMap2d v_map = VoxelMap2d()
    #image  constants
    cdef int rows, cols
    rows = <int>(max_range/resolution + 1)
    cols = <int>(max_range/resolution + 1)

    cdef int[:] raw_bev_img = np.zeros(rows * cols, dtype=np.int32)

    cdef Voxel2d v_xy
    for i in range(num_points):
        v_xy = Voxel2d(cloud_xy_voxels[i][0], cloud_xy_voxels[i][1])
        v_map[v_xy] = v_map[v_xy] + 1

    # Iterate through unique voxels in the map and fill the image
    cdef VoxelMap2d.iterator it = v_map.begin()
    cdef pair[Voxel2d, int] entry
    cdef int index
    while it != v_map.end():
        entry = deref(it)
        v_xy = entry.first
        # Check bounds before indexing
        if v_xy.x >= 0 and v_xy.x < cols and v_xy.y >= 0 and v_xy.y < rows:
            index = cols * v_xy.y + v_xy.x
            raw_bev_img[index] = entry.second
        inc(it)

    return np.array(raw_bev_img).reshape((rows, cols))
