from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool as bool_t
from libcpp.utility cimport pair

cdef extern from "<functional>" namespace "std":
    cdef cppclass hash[T]:
        pass

cdef extern from *:
    """
    #include <unordered_map>
    #include <functional>

    struct Voxel2d {
        int x;
        int y;

        Voxel2d(): x(0), y(0){}
        
        Voxel2d(int xVal, int yVal){
            x = xVal;
            y = yVal;
        }

        bool operator==(const Voxel2d& other) const {
            return x == other.x && y == other.y;
        }
    };

    struct VoxelHash {
        std::size_t operator()(const Voxel2d& c) const noexcept {
            // A decent hash combination for two ints
            return std::hash<int>()(c.x) ^ (std::hash<int>()(c.y) << 1);
        }
    };

    typedef std::unordered_map<Voxel2d, int, VoxelHash> VoxelMap2d;
    """
    cdef cppclass Voxel2d:
        int x
        int y
        Voxel2d() except +
        Voxel2d(int x, int y) except +
        bool_t operator==(Voxel2d& other) const

    cdef cppclass VoxelHash:
        VoxelHash() except +

    cdef cppclass VoxelMap2d:
        VoxelMap2d() except +
        int& operator[](Voxel2d& key)
        size_t size() const
        cppclass iterator:
            iterator() except +
            iterator(iterator&) except +
            iterator& operator=(iterator&)
            bint operator==(iterator&)
            bint operator!=(iterator&)
            iterator& operator++()
            pair[Voxel2d, int]& operator*()
        iterator begin()
        iterator end()
