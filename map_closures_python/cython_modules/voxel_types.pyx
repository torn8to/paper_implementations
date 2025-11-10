# cython_modules/voxel_types.pxd
# distutils: language = c++
# cython: language_level=3

from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from .voxel_types cimport Voxel2d as cVoxel2d, VoxelMap2d as cVoxelMap2d

class Voxel2dPy:
    """Python wrapper for the C++ Voxel2d struct"""
    cdef cVoxel2d cpp_voxel

    def __cinit__(self, int x=0, int y=0):
        self.cpp_voxel = cVoxel2d(x, y)

    @property
    def x(self): return self.cpp_voxel.x

    @property
    def y(self): return self.cpp_voxel.y

    def __eq__(self, other: Voxel2dPy):
        if not isinstance(other, Voxel2dPy):
            return False
        return self.cpp_voxel == other.cpp_voxel

    def __repr__(self):
        return f"Voxel2d(x={self.x}, y={self.y})"


class VoxelMap2dPy:
    """Python wrapper for the C++ unordered_map<Voxel2d, int, VoxelHash> 
        
    Atrributes
    ----------
    cpp_map: VoxelMap2d c++ its std::unordered_map<Voxel2d, int>
    """
    cdef cVoxelMap2d cpp_map

    def __cinit__(self):
        self.cpp_map = cVoxelMap2d()

    def __getitem__(self, Voxel2dPy key):
        return self.cpp_map[key.cpp_voxel]

    def __setitem__(self, Voxel2dPy key, int value):
        self.cpp_map[key.cpp_voxel] = value

    def __len__(self):
        return self.cpp_map.size()

    def __repr__(self):
        return f"VoxelMap2d(size={len(self)})"


