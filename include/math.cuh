#ifndef LBVH_MATH_CUH
#define LBVH_MATH_CUH
#include "types.h"

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

__device__ __host__ float3 operator*(float scalar, const float3& vec) {
    return make_float3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
}

__device__ __host__ float3 operator*(const float3& vec, float scalar) {
    return scalar * vec;  // Reuse the previously defined operator
}



__device__ __host__ float3 cross(const float3 &u, const float3 &v) {
        float3 w;
        w.x = u.y * v.z - u.z * v.y;
        w.y = u.z * v.x - u.x * v.z;
        w.z = u.x * v.y - u.y * v.x;
        return w;
    }
__device__ __host__ float dot(const float3 &u, const float3 &v) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    }


#endif