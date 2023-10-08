#ifndef TYPES_H
#define TYPES_H

// BUFFER SIZE : buffer size for query
#define BUFFER_SIZE 64
// FACE SIZE : buffer size for adj faces
#define FACES_SIZE 64

// Operator overload for printing float3 using cout
std::ostream& operator<<(std::ostream& os, const float3& vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

template <typename T>
struct Vertex
{
    T x, y, z;

    inline __device__ __host__ T *data_ptr() { return &x; }
};

template <typename T>
struct Face
{
    T i, j, k;

    inline __device__ __host__ T *data_ptr() { return &i; }
};

template <typename T>
struct Triangle
{
    T v0, v1, v2; // 삼각형의 3개 꼭짓점

    inline __device__ __host__ T *data_ptr() { return &v0; }
};

struct aabb_getter
{
    __device__ lbvh::aabb<float> operator()(const Triangle<float3> &tri) const noexcept
    {
        lbvh::aabb<float> box;

        // 각 꼭짓점에서 x, y, z의 최소 및 최대 값을 찾습니다.
        box.lower.x = min(tri.v0.x, min(tri.v1.x, tri.v2.x));
        box.lower.y = min(tri.v0.y, min(tri.v1.y, tri.v2.y));
        box.lower.z = min(tri.v0.z, min(tri.v1.z, tri.v2.z));

        box.upper.x = max(tri.v0.x, max(tri.v1.x, tri.v2.x));
        box.upper.y = max(tri.v0.y, max(tri.v1.y, tri.v2.y));
        box.upper.z = max(tri.v0.z, max(tri.v1.z, tri.v2.z));

        return box;
    }
};

#endif