#ifndef LBVH_TRI_TRI_INTERSECT_CUH
#define LBVH_TRI_TRI_INTERSECT_CUH
#include "types.cuh"
#include "math.cuh"
#include <thrust/for_each.h>
#include <vector>
#include <cmath>

namespace selfx{

struct double3{
  double x,y,z;
};

// double3 덧셈 연산자 오버로딩
__device__ __host__
double3 operator+(const double3 &a, const double3 &b) {
  double3 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

// double3 뺄셈 연산자 오버로딩
__device__ __host__
double3 operator-(const double3 &a, const double3 &b) {
  double3 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;
  return result;
}

__device__ __host__
double3 operator*(const double3 &v, double scalar) {
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

// double와 double3 곱하기 (순서가 바뀐 경우)
__device__ __host__
double3 operator*(double scalar, const double3 &v) {
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

__device__ __host__
inline double3 cross_product_double3(const double3 &v1, const double3 &v2){
  double3 result;
  result.x = v1.y * v2.z - v1.z * v2.y;
  result.y = v1.z * v2.x - v1.x * v2.z;
  result.z = v1.x * v2.y - v1.y * v2.x;
  return result;
}

__device__ __host__
inline double dot_product_double3(const double3 &v1, const double3 &v2){
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// float3를 double3로 변환
__device__ __host__
inline double3 to_double3(const float3 &f) {
    return {static_cast<double>(f.x), static_cast<double>(f.y), static_cast<double>(f.z)};
}

// double3를 float3로 변환
__device__ __host__
inline float3 to_float3(const double3 &d) {
    return {static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z)};
}

// --------------------------------double3

__device__ __host__
inline bool check_if_source_and_target_is_same(float3 source, float3 target){
  return(source.x == target.x) && (source.y == target.y) && (source.z == target.z);
}

__device__ __host__
inline float compute_barycentric_coordinate(const float3 &a, const float3 &b, const float3 &c, const float3 &point) {
    float detT = ((b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y));
    return ((b.y - c.y)*(point.x - c.x) + (c.x - b.x)*(point.y - c.y)) / detT;
}

__device__ __host__
inline bool is_inside_triangle(float alpha, float beta, float gamma) {
    return (alpha > 0 && alpha < 1 && beta > 0 && beta < 1 && gamma > 0 && gamma < 1);
}

__device__ __host__
inline bool check_intersection_in_open_region(const float3 &p1, const float3 &q1, const float3 &r1,
                                   const float3 &p2, const float3 &q2, const float3 &r2,
                                   const float3 &source, const float3 &target) {
    // Compute barycentric coordinates for the first triangle
    float alpha_source_1 = compute_barycentric_coordinate(p1, q1, r1, source);
    float beta_source_1 = compute_barycentric_coordinate(q1, r1, p1, source);
    float gamma_source_1 = 1.0f - alpha_source_1 - beta_source_1;

    float alpha_target_1 = compute_barycentric_coordinate(p1, q1, r1, target);
    float beta_target_1 = compute_barycentric_coordinate(q1, r1, p1, target);
    float gamma_target_1 = 1.0f - alpha_target_1 - beta_target_1;

    // Compute barycentric coordinates for the second triangle
    float alpha_source_2 = compute_barycentric_coordinate(p2, q2, r2, source);
    float beta_source_2 = compute_barycentric_coordinate(q2, r2, p2, source);
    float gamma_source_2 = 1.0f - alpha_source_2 - beta_source_2;

    float alpha_target_2 = compute_barycentric_coordinate(p2, q2, r2, target);
    float beta_target_2 = compute_barycentric_coordinate(q2, r2, p2, target);
    float gamma_target_2 = 1.0f - alpha_target_2 - beta_target_2;

    // Check if the source and target are inside the triangles
    bool isIntersectionInside = is_inside_triangle(alpha_source_1, beta_source_1, gamma_source_1) &&
                                is_inside_triangle(alpha_target_1, beta_target_1, gamma_target_1) &&
                                is_inside_triangle(alpha_source_2, beta_source_2, gamma_source_2) &&
                                is_inside_triangle(alpha_target_2, beta_target_2, gamma_target_2);

    return isIntersectionInside;
}

__device__ __host__
inline bool check_if_vertex(const float3 &point, const float3 &vertex, float epsilon) {
    //float epsilon = 1e-4;  // Define a small tolerance value
    return std::abs(point.x - vertex.x) < epsilon &&
           std::abs(point.y - vertex.y) < epsilon &&
           std::abs(point.z - vertex.z) < epsilon;
}

__device__ __host__ float largest_distance(const float3 &a, const float3 &b){
    float dx = std::abs(b.x - a.x);
    float dy = std::abs(b.y - a.y);
    float dz = std::abs(b.z - a.z);
    float largest_dist = -std::numeric_limits<float>::infinity();
    if (dx > largest_dist) largest_dist = dx;
    if (dy > largest_dist) largest_dist = dy;
    if (dz > largest_dist) largest_dist = dz;
    return largest_dist;
    //return sqrt(dx * dx + dy * dy + dz * dz);
}

__device__ __host__ float distance(const float3 &a, const float3 &b){
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}


// return : if two faces share at least one vertex
__device__ __host__
inline bool check_if_shared_vertex(const float3 &p1, const float3 &q1, const float3 &r1,
                   const float3 &p2, const float3 &q2, const float3 &r2){
    return (p1 == p2) || (p1 == q2) || (p1 == r2) || (q1 == p2) || (q1 == q2) || (q1 == r2)
            || (r1 == p2) || (r1 == q2) || (r1 == r2);
}

__device__ __host__
inline bool check_vertices(const float3 &p1, const float3 &q1, const float3 &r1,
                   const float3 &p2, const float3 &q2, const float3 &r2,
                   const float3 &source, const float3 &target, float epsilon) {
    // Check if source and target correspond to the same vertex 이거 사용!!!!
    // bool isSameVertex = check_if_vertex(source, target);

    // if (isSameVertex) {
    //     // Check if the source and target (which are the same vertex) are vertices of any triangle
    //     bool isVertexOfTriangle1 = check_if_vertex(source, p1) || check_if_vertex(source, q1) || check_if_vertex(source, r1);
    //     bool isVertexOfTriangle2 = check_if_vertex(source, p2) || check_if_vertex(source, q2) || check_if_vertex(source, r2);

    //     return isVertexOfTriangle1 || isVertexOfTriangle2;
    // }
    // return false;

    float3 midpoint;
    midpoint.x = (source.x + target.x) / 2;
    midpoint.y = (source.y + target.y) / 2;
    midpoint.z = (source.z + target.z) / 2;
    // Check if the source and target (which are the same vertex) are vertices of any triangle
    bool isVertexOfTriangle1 = check_if_vertex(midpoint, p1, epsilon) || check_if_vertex(midpoint, q1, epsilon) || check_if_vertex(midpoint, r1, epsilon);
    bool isVertexOfTriangle2 = check_if_vertex(midpoint, p2, epsilon) || check_if_vertex(midpoint, q2, epsilon) || check_if_vertex(midpoint, r2, epsilon);

    return isVertexOfTriangle1 || isVertexOfTriangle2;


    // // Check if the source and target are vertices of the first triangle
    // bool isSourceVertex1 = check_if_vertex(source, p1) || check_if_vertex(source, q1) || check_if_vertex(source, r1);
    // bool isTargetVertex1 = check_if_vertex(target, p1) || check_if_vertex(target, q1) || check_if_vertex(target, r1);

    // // Check if the source and target are vertices of the second triangle
    // bool isSourceVertex2 = check_if_vertex(source, p2) || check_if_vertex(source, q2) || check_if_vertex(source, r2);
    // bool isTargetVertex2 = check_if_vertex(target, p2) || check_if_vertex(target, q2) || check_if_vertex(target, r2);

    // return (isSourceVertex1 && isTargetVertex1) || (isSourceVertex2 && isTargetVertex2);
}



__device__ __host__
inline float ORIENT_2D(const float2& a, const float2& b, const float2& c)
{
    return (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);
}

__device__ __host__
bool INTERSECTION_TEST_EDGE(
  const float2 & P1, const float2 & Q1, const float2 & R1,  
  const float2 & P2, const float2 & Q2, const float2 & R2
)
{
  if (ORIENT_2D(R2,P2,Q1) >= 0.0) {
    if (ORIENT_2D(P1,P2,Q1) >= 0.0) {
        if (ORIENT_2D(P1,Q1,R2) >= 0.0) return true;
        else return false;} else { 
      if (ORIENT_2D(Q1,R1,P2) >= 0.0){ 
  if (ORIENT_2D(R1,P1,P2) >= 0.0) return true; else return false;} 
      else return false; } 
  } else {
    if (ORIENT_2D(R2,P2,R1) >= 0.0) {
      if (ORIENT_2D(P1,P2,R1) >= 0.0) {
  if (ORIENT_2D(P1,R1,R2) >= 0.0) return true;  
  else {
    if (ORIENT_2D(Q1,R1,R2) >= 0.0) return true; else return false;}}
      else  return false; }
    else return false; }
}

__device__ __host__
bool INTERSECTION_TEST_VERTEX(
  const float2 & P1, const float2 & Q1, const float2 & R1,  
  const float2 & P2, const float2 & Q2, const float2 & R2
)
{
  if (ORIENT_2D(R2,P2,Q1) >= 0.0)
    if (ORIENT_2D(R2,Q2,Q1) <= 0.0)
      if (ORIENT_2D(P1,P2,Q1) > 0.0) {
        if (ORIENT_2D(P1,Q2,Q1) <= 0.0) return true;
        else return false;} 
      else {
        if (ORIENT_2D(P1,P2,R1) >= 0.0)
          if (ORIENT_2D(Q1,R1,P2) >= 0.0) return true; 
          else return false;
        else return false;
      }
      else 
        if (ORIENT_2D(P1,Q2,Q1) <= 0.0)
          if (ORIENT_2D(R2,Q2,R1) <= 0.0)
            if (ORIENT_2D(Q1,R1,Q2) >= 0.0) return true; 
            else return false;
          else return false;
        else return false;
      else
        if (ORIENT_2D(R2,P2,R1) >= 0.0) 
          if (ORIENT_2D(Q1,R1,R2) >= 0.0)
            if (ORIENT_2D(P1,P2,R1) >= 0.0) return true;
            else return false;
          else 
            if (ORIENT_2D(Q1,R1,Q2) >= 0.0) {
              if (ORIENT_2D(R2,R1,Q2) >= 0.0) return true; 
              else return false; 
            }
        else return false; 
  else  return false; 
}

__device__ __host__
bool ccw_tri_tri_intersection_2d(
  const float2 &p1, const float2 &q1, const float2 &r1,
  const float2 &p2, const float2 &q2, const float2 &r2)
  {
  if ( ORIENT_2D(p2,q2,p1) >= 0.0 ) {
    if ( ORIENT_2D(q2,r2,p1) >= 0.0 ) {
      if ( ORIENT_2D(r2,p2,p1) >= 0.0 ) return true;
      else return INTERSECTION_TEST_EDGE(p1,q1,r1,p2,q2,r2);
    } else {  
      if ( ORIENT_2D(r2,p2,p1) >= 0.0 ) 
      return INTERSECTION_TEST_EDGE(p1,q1,r1,r2,p2,q2);
      else return INTERSECTION_TEST_VERTEX(p1,q1,r1,p2,q2,r2);}}
  else {
    if ( ORIENT_2D(q2,r2,p1) >= 0.0 ) {
      if ( ORIENT_2D(r2,p2,p1) >= 0.0 ) 
        return INTERSECTION_TEST_EDGE(p1,q1,r1,q2,r2,p2);
      else  return INTERSECTION_TEST_VERTEX(p1,q1,r1,q2,r2,p2);}
    else return INTERSECTION_TEST_VERTEX(p1,q1,r1,r2,p2,q2);}
};

__device__ __host__
bool tri_tri_overlap_test_2d(
  const float2 &p1, const float2 &q1, const float2 &r1,
  const float2 &p2, const float2 &q2, const float2 &r2) 
{
  if ( ORIENT_2D(p1,q1,r1) < 0.0)
    if ( ORIENT_2D(p2,q2,r2) < 0.0)
      return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,r2,q2);
    else
      return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,q2,r2);
  else
    if ( ORIENT_2D(p2,q2,r2) < 0.0 )
      return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,r2,q2);
    else
      return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,q2,r2);
};



__device__ __host__
bool coplanar_tri_tri3d(
    const float3 &p1, const float3 &q1, const float3 &r1,
    const float3 &p2, const float3 &q2, const float3 &r2,
    const float3 &normal_1)
{
    float2 P1, Q1, R1;
    float2 P2, Q2, R2;

    float n_x = fabs(normal_1.x);
    float n_y = fabs(normal_1.y);
    float n_z = fabs(normal_1.z);

    /* Projection of the triangles in 3D onto 2D such that the area of
    the projection is maximized. */

    if (n_x > n_z && n_x >= n_y) {
        // Project onto plane YZ
        P1 = make_float2(q1.z, q1.y);
        Q1 = make_float2(p1.z, p1.y);
        R1 = make_float2(r1.z, r1.y);

        P2 = make_float2(q2.z, q2.y);
        Q2 = make_float2(p2.z, p2.y);
        R2 = make_float2(r2.z, r2.y);
    } else if (n_y > n_z && n_y >= n_x) {
        // Project onto plane XZ
        P1 = make_float2(q1.x, q1.z);
        Q1 = make_float2(p1.x, p1.z);
        R1 = make_float2(r1.x, r1.z);

        P2 = make_float2(q2.x, q2.z);
        Q2 = make_float2(p2.x, p2.z);
        R2 = make_float2(r2.x, r2.z);
    } else {
        // Project onto plane XY
        P1 = make_float2(p1.x, p1.y);
        Q1 = make_float2(q1.x, q1.y);
        R1 = make_float2(r1.x, r1.y);

        P2 = make_float2(p2.x, p2.y);
        Q2 = make_float2(q2.x, q2.y);
        R2 = make_float2(r2.x, r2.y);
    }

    return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);  // Make sure this function is also adapted for float2.
}

// double
// __device__ __host__
// bool CONSTRUCT_INTERSECTION(
//   const float3 &p1, const float3 &q1, const float3 &r1,
//   const float3 &p2, const float3 &q2, const float3 &r2,
//         float3  &source, float3 &target,
//   const float3 &N1,const float3 &N2)
// {
//   //   float3 v,v1,v2,N;

// //   v1=q1-p1;
// //   v2=r2-p1
//   double3 dp1 = to_double3(p1), dq1 = to_double3(q1), dr1 = to_double3(r1);
//   double3 dp2 = to_double3(p2), dq2 = to_double3(q2), dr2 = to_double3(r2);
//   double3 dN1 = to_double3(N1), dN2 = to_double3(N2);
//   double3 dv, dv1, dv2, dN;
  
//   dv1 = dq1 - dp1;
//   dv2 = dr2 - dp1;
//   dN = cross_product_double3(dv1, dv2);
//   dv = dp2 - dp1;

//   if (dot_product_double3(dv,dN) > 0.0) {
//     dv1=dr1-dp1;
//     //N=v1.cross(v2);
//     dN = cross_product_double3(dv1, dv2);
//     if (dot_product_double3(dv, dN) <= 0.0) {
//       dv2=dq2-dp1;
//       //N=v1.cross(v2);
//       dN = cross_product_double3(dv1,dv2);
//       if (dot_product_double3(dv, dN) > 0.0) {
//         dv1=dp1-dp2;
//         dv2=dp1-dr1;
//         //Scalar alpha = v1.dot(N2) / v2.dot(N2);
//         double alpha = dot_product_double3(dv1, dN2) / dot_product_double3(dv2, dN2);
//         dv1=dv2*alpha;
//         source=to_float3(dp1-dv1);
//         dv1=dp2-dp1;
//         dv2=dp2-dr2;
//         //alpha = v1.dot(N1) / v2.dot(N1);
//         alpha = dot_product_double3(dv1, dN1) / dot_product_double3(dv2, dN1);
//         dv1=dv2*alpha;
//         target=to_float3(dp2-dv1);
//         return true;
//       } else { 
//         dv1=dp2-dp1;
//         dv2=dp2-dq2;
//         //Scalar alpha = v1.dot(N1) / v2.dot(N1);
//         double alpha = dot_product_double3(dv1, dN1) / dot_product_double3(dv2, dN1);
//         dv1=dv2*alpha;
//         source=to_float3(dp2-dv1);
//         dv1=dp2-dp1;
//         dv2=dp2-dr2;
//         //alpha = v1.dot(N1) / v2.dot(N1);
//         alpha = dot_product_double3(dv1, dN1) / dot_product_double3(dv2, dN1);
//         dv1=dv2*alpha;
//         target=to_float3(dp2-dv1);
//         return true;
//       } 
//     } else {
//       return false;
//     } 
//   } else { 
//     dv2=dq2-dp1;
//     //N=v1.cross(v2);
//     dN = cross_product_double3(dv1, dv2);
//     if (dot_product_double3(dv, dN) < 0.0) {
//       return false;
//     } else {
//       dv1=dr1-dp1;
//       //N=v1.cross(v2);
//       dN = cross_product_double3(dv1, dv2);
//       if (dot_product_double3(dv,dN) >= 0.0) { 
//         dv1=dp1-dp2;
//         dv2=dp1-dr1;
//         //Scalar alpha = v1.dot(N2) / v2.dot(N2);
//         double alpha = dot_product_double3(dv1, dN2) / dot_product_double3(dv2, dN2);
//         dv1=dv2*alpha;
//         source=to_float3(dp1-dv1);
//         dv1=dp1-dp2;
//         dv2=dp1-dq1;
//         //alpha = v1.dot(N2) / v2.dot(N2);
//         alpha = dot_product_double3(dv1, dN2) / dot_product_double3(dv2, dN2);
//         dv1=dv2*alpha;
//         target=to_float3(dp1-dv1);
//         return true; 
//       } else { 
//         dv1=dp2-dp1;
//         dv2=dp2-dq2;
//         //Scalar alpha = v1.dot(N1) / v2.dot(N1);
//         double alpha = dot_product_double3(dv1, dN1) / dot_product_double3(dv2, dN1);
//         dv1=dv2*alpha;
//         source=to_float3(dp2-dv1);
//         dv1=dp1-dp2;
//         dv2=dp1-dq1;
//         //alpha = v1.dot(N2) / v2.dot(N2);
//         alpha = dot_product_double3(dv1, dN2) / dot_product_double3(dv2, dN2);
//         dv1=dv2*alpha;
//         target=to_float3(dp1-dv1);

//         //printf("source %f %f %f, target %f %f %f\n", source.x, source.y, source.z, target.x, target.y, target.z);
//         return true;
//     }}}
// }

__device__ __host__
bool CONSTRUCT_INTERSECTION(
  const float3 &p1, const float3 &q1, const float3 &r1,
  const float3 &p2, const float3 &q2, const float3 &r2,
        float3  &source, float3 &target,
  const float3 &N1,const float3 &N2)
{

  float3 v,v1,v2,N;

  v1=q1-p1;
  v2=r2-p1;
  //N=v1.cross(v2);
  N = cross(v1, v2);
  v=p2-p1;
  if (dot(v,N) > 0.0) {
    v1=r1-p1;
    //N=v1.cross(v2);
    N = cross(v1, v2);
    if (dot(v, N) <= 0.0) {
      v2=q2-p1;
      //N=v1.cross(v2);
      N = cross(v1,v2);
      if (dot(v, N) > 0.0) {
        v1=p1-p2;
        v2=p1-r1;
        //Scalar alpha = v1.dot(N2) / v2.dot(N2);
        float alpha = dot(v1, N2) / dot(v2, N2);
        v1=v2*alpha;
        source=p1-v1;
        v1=p2-p1;
        v2=p2-r2;
        //alpha = v1.dot(N1) / v2.dot(N1);
        alpha = dot(v1, N1) / dot(v2, N1);
        v1=v2*alpha;
        target=p2-v1;
        return true;
      } else { 
        v1=p2-p1;
        v2=p2-q2;
        //Scalar alpha = v1.dot(N1) / v2.dot(N1);
        float alpha = dot(v1, N1) / dot(v2, N1);
        v1=v2*alpha;
        source=p2-v1;
        v1=p2-p1;
        v2=p2-r2;
        //alpha = v1.dot(N1) / v2.dot(N1);
        alpha = dot(v1, N1) / dot(v2, N1);
        v1=v2*alpha;
        target=p2-v1;
        return true;
      } 
    } else {
      return false;
    } 
  } else { 
    v2=q2-p1;
    //N=v1.cross(v2);
    N = cross(v1, v2);
    if (dot(v, N) < 0.0) {
      return false;
    } else {
      v1=r1-p1;
      //N=v1.cross(v2);
      N = cross(v1, v2);
      if (dot(v,N) >= 0.0) { 
        v1=p1-p2;
        v2=p1-r1;
        //Scalar alpha = v1.dot(N2) / v2.dot(N2);
        float alpha = dot(v1, N2) / dot(v2, N2);
        v1=v2*alpha;
        source=p1-v1;
        v1=p1-p2;
        v2=p1-q1;
        //alpha = v1.dot(N2) / v2.dot(N2);
        alpha = dot(v1, N2) / dot(v2, N2);
        v1=v2*alpha;
        target=p1-v1 ;
        return true; 
      } else { 
        v1=p2-p1 ;
        v2=p2-q2 ;
        //Scalar alpha = v1.dot(N1) / v2.dot(N1);
        float alpha = dot(v1, N1) / dot(v2, N1);
        v1=v2*alpha;
        source=p2-v1;
        v1=p1-p2;
        v2=p1-q1;
        //alpha = v1.dot(N2) / v2.dot(N2);
        alpha = dot(v1, N2) / dot(v2, N2);
        v1=v2*alpha;
        target=p1-v1;

        //printf("source %f %f %f, target %f %f %f\n", source.x, source.y, source.z, target.x, target.y, target.z);
        return true;
    }}}
}
    

    __device__ __host__
    bool TRI_TRI_INTER_3D(
        const float3 &p1, const float3 &q1, const float3 &r1,
        const float3 &p2, const float3 &q2, const float3 &r2,
        float dp2, float dq2, float dr2,
        bool & coplanar,
        float3 &source, float3 &target,
        const float3 &N1,const float3 &N2
        )
    {
    if (dp2 > 0.0) { 
     if (dq2 > 0.0) return CONSTRUCT_INTERSECTION(p1,r1,q1,r2,p2,q2,source,target,N1,N2);
     else if (dr2 > 0.0) return CONSTRUCT_INTERSECTION(p1,r1,q1,q2,r2,p2,source,target,N1,N2);
          else return CONSTRUCT_INTERSECTION(p1,q1,r1,p2,q2,r2,source,target,N1,N2); }
    else if (dp2 < 0.0) { 
      if (dq2 < 0.0) return CONSTRUCT_INTERSECTION(p1,q1,r1,r2,p2,q2,source,target,N1,N2);
      else if (dr2 < 0.0) return CONSTRUCT_INTERSECTION(p1,q1,r1,q2,r2,p2,source,target,N1,N2);
          else return CONSTRUCT_INTERSECTION(p1,r1,q1,p2,q2,r2,source,target,N1,N2);
    } else { 
      if (dq2 < 0.0) { 
        if (dr2 >= 0.0)  return CONSTRUCT_INTERSECTION(p1,r1,q1,q2,r2,p2,source,target,N1,N2);
        else return CONSTRUCT_INTERSECTION(p1,q1,r1,p2,q2,r2,source,target,N1,N2);
    } 
    else if (dq2 > 0.0) {
      if (dr2 > 0.0) return CONSTRUCT_INTERSECTION(p1,r1,q1,p2,q2,r2,source,target,N1,N2);
      else  return CONSTRUCT_INTERSECTION(p1,q1,r1,q2,r2,p2,source,target,N1,N2);
    } 
    else  {
      if (dr2 > 0.0) return CONSTRUCT_INTERSECTION(p1,q1,r1,r2,p2,q2,source,target,N1,N2);
      else if (dr2 < 0.0) return CONSTRUCT_INTERSECTION(p1,r1,q1,r2,p2,q2,source,target,N1,N2);
      else { 
        coplanar = true;
        // we add this coplanar logic in another step
        return false;
        //return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1);
     }
  }
    }
  }


__device__ __host__
bool tri_tri_intersection_test_3d(
    const float3 & p1, const float3 & q1, const float3 & r1, 
    const float3 & p2, const float3 & q2, const float3 & r2,
    bool & coplanar,
    float3 & source, float3 & target )
{
  //using Scalar = typename DerivedP1::Scalar;
  //using RowVector3D = typename Eigen::Matrix<Scalar, 1, 3>;

  float dp1, dq1, dr1, dp2, dq2, dr2;
  float3 v1, v2;//, v;
  float3 N1, N2;//, N;

  // source = make_float3(1,1,1);
  // target = make_float3(-1,-1,-1);
  //float alpha;
  // Compute distance signs  of p1, q1 and r1 
  // to the plane of triangle(p2,q2,r2)

  v1=p2-r2;
  v2=q2-r2;
  N2=cross(v1,v2);


  v1=p1-r2;
  //dp1 = v1.dot(N2);
  dp1 = dot(v1, N2);
  v1=q1-r2;
  //dq1 = v1.dot(N2);
  dq1 = dot(v1, N2);
  v1=r1-r2;
  //dr1 = v1.dot(N2);
  dr1 = dot(v1, N2);
  
  coplanar = false;

  if (((dp1 * dq1) > 0.0) && ((dp1 * dr1) > 0.0))  return false; 

  // Compute distance signs  of p2, q2 and r2 
  // to the plane of triangle(p1,q1,r1)

  
  v1=q1-p1;
  v2=r1-p1;
  //N1=v1.cross(v2);
  N1 = cross(v1, v2);

  v1=p2-r1;
  //dp2 = v1.dot(N1);
  dp2 = dot(v1, N1);
  v1=q2-r1;
  //dq2 = v1.dot(N1);
  dq2 = dot(v1, N1);
  v1=r2-r1;
  //dr2 = v1.dot(N1);
  dr2 = dot(v1, N1);
  
  if (((dp2 * dq2) > 0.0) && ((dp2 * dr2) > 0.0)) return false;

  // Permutation in a canonical form of T1's vertices
  if (dp1 > 0.0) {
    if (dq1 > 0.0) return TRI_TRI_INTER_3D(r1,p1,q1,p2,r2,q2,dp2,dr2,dq2,coplanar,source,target,N1,N2);
    else if (dr1 > 0.0) return TRI_TRI_INTER_3D(q1,r1,p1,p2,r2,q2,dp2,dr2,dq2,coplanar,source,target,N1,N2);
  
    else return TRI_TRI_INTER_3D(p1,q1,r1,p2,q2,r2,dp2,dq2,dr2,coplanar,source,target,N1,N2);
  } else if (dp1 < 0.0) {
    if (dq1 < 0.0) return TRI_TRI_INTER_3D(r1,p1,q1,p2,q2,r2,dp2,dq2,dr2,coplanar,source,target,N1,N2);
    else if (dr1 < 0.0) return TRI_TRI_INTER_3D(q1,r1,p1,p2,q2,r2,dp2,dq2,dr2,coplanar,source,target,N1,N2);
    else return TRI_TRI_INTER_3D(p1,q1,r1,p2,r2,q2,dp2,dr2,dq2,coplanar,source,target,N1,N2);
  } else {
    if (dq1 < 0.0) {
      if (dr1 >= 0.0) return TRI_TRI_INTER_3D(q1,r1,p1,p2,r2,q2,dp2,dr2,dq2,coplanar,source,target,N1,N2);
      else return TRI_TRI_INTER_3D(p1,q1,r1,p2,q2,r2,dp2,dq2,dr2,coplanar,source,target,N1,N2);
    }
    else if (dq1 > 0.0) {
      if (dr1 > 0.0) return TRI_TRI_INTER_3D(p1,q1,r1,p2,r2,q2,dp2,dr2,dq2,coplanar,source,target,N1,N2);
      else return TRI_TRI_INTER_3D(q1,r1,p1,p2,q2,r2,dp2,dq2,dr2,coplanar,source,target,N1,N2);
    }
    else  {
      if (dr1 > 0.0) return TRI_TRI_INTER_3D(r1,p1,q1,p2,q2,r2,dp2,dq2,dr2,coplanar,source,target,N1,N2);
      else if (dr1 < 0.0) return TRI_TRI_INTER_3D(r1,p1,q1,p2,r2,q2,dp2,dr2,dq2,coplanar,source,target,N1,N2);
      else {
        // triangles are co-planar

        coplanar = true;
        // we add this coplanar logic in another step
        return false;
        //return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1);
      }
    }
  }
};

// // double
// __device__ __host__ bool is_coplanar(
//     const float3 & p1, const float3 & q1, const float3 & r1, 
//     const float3 & p2, const float3 & q2, const float3 & r2, float eps)
// {
//     double dp1, dq1, dr1, dp2, dq2, dr2;
//     double3 v1, v2;
//     double3 N1, N2;
//     eps = 1e-7;

//     // Convert float3 to double3
//     double3 dp1_3 = to_double3(p1);
//     double3 dq1_3 = to_double3(q1);
//     double3 dr1_3 = to_double3(r1);
//     double3 dp2_3 = to_double3(p2);
//     double3 dq2_3 = to_double3(q2);
//     double3 dr2_3 = to_double3(r2);

//     // Compute distance signs of p1, q1, r1 to the plane of triangle (p2, q2, r2)
//     v1 = dp2_3 - dr2_3;
//     v2 = dq2_3 - dr2_3;
//     N2 = cross_product_double3(v1, v2);

//     v1 = dp1_3 - dr2_3;
//     dp1 = dot_product_double3(v1, N2);
//     v1 = dq1_3 - dr2_3;
//     dq1 = dot_product_double3(v1, N2);
//     v1 = dr1_3 - dr2_3;
//     dr1 = dot_product_double3(v1, N2);

//     if (fabs(dp1) >= eps || fabs(dq1) >= eps || fabs(dr1) >= eps) return false;

//     // Compute distance signs of p2, q2, r2 to the plane of triangle (p1, q1, r1)
//     v1 = dq1_3 - dp1_3;
//     v2 = dr1_3 - dp1_3;
//     N1 = cross_product_double3(v1, v2);

//     v1 = dp2_3 - dr1_3;
//     dp2 = dot_product_double3(v1, N1);
//     v1 = dq2_3 - dr1_3;
//     dq2 = dot_product_double3(v1, N1);
//     v1 = dr2_3 - dr1_3;
//     dr2 = dot_product_double3(v1, N1);

//     if (fabs(dp2) >= eps || fabs(dq2) >= eps || fabs(dr2) >= eps) return false;

//     printf("dp1 %lf dq1 %lf dr1 %lf dp2 %lf dq2 %lf dr2 %lf\n", dp1, dq1, dr1, dp2, dq2, dr2);
//     return true;
// }


// //float
// // if two triangles are coplanar
// __device__ __host__ bool is_coplanar(
//     const float3 & p1, const float3 & q1, const float3 & r1, 
//     const float3 & p2, const float3 & q2, const float3 & r2, float eps)
//     {
//       float dp1, dq1, dr1, dp2, dq2, dr2;
//       float3 v1, v2;//, v;
//       float3 N1, N2;//, N;
//       eps = 1e-7;
//       //float alpha;
//       // Compute distance signs  of p1, q1 and r1 
//       // to the plane of triangle(p2,q2,r2)

//       v1=p2-r2;
//       v2=q2-r2;
//       N2=cross(v1,v2);


//       v1=p1-r2;
//       //dp1 = v1.dot(N2);
//       dp1 = dot(v1, N2);
//       v1=q1-r2;
//       //dq1 = v1.dot(N2);
//       dq1 = dot(v1, N2);
//       v1=r1-r2;
//       //dr1 = v1.dot(N2);
//       dr1 = dot(v1, N2);
      

//       //if (((dp1 * dq1) > 0.0) && ((dp1 * dr1) > 0.0))  return false;
//       //if((dp1 >= eps) || (dq1 >= eps) || (dr1 >= eps)) return false;
//       if(fabs(dp1) >= eps || fabs(dq1) >= eps || fabs(dr1) >= eps) return false;


//       // Compute distance signs  of p2, q2 and r2 
//       // to the plane of triangle(p1,q1,r1)

      
//       v1=q1-p1;
//       v2=r1-p1;
//       //N1=v1.cross(v2);
//       N1 = cross(v1, v2);

//       v1=p2-r1;
//       //dp2 = v1.dot(N1);
//       dp2 = dot(v1, N1);
//       v1=q2-r1;
//       //dq2 = v1.dot(N1);
//       dq2 = dot(v1, N1);
//       v1=r2-r1;
//       //dr2 = v1.dot(N1);
//       dr2 = dot(v1, N1);
      
//       //if (((dp2 * dq2) > 0.0) && ((dp2 * dr2) > 0.0)) return false;
//       //if((dp2 >= eps) || (dq2 >= eps) || (dr2 >= eps)) return false;
//       if(fabs(dp2) >= eps || fabs(dq2) >= eps || fabs(dr2) >= eps) return false;

//       printf("dp1 %f dq1 %f dr1 %f dp2 %f dq2 %f dr2 %f\n", dp1, dq1, dr1, dp2, dq2, dr2);
//       return true;
//     }

// custom
__device__ __host__ bool is_coplanar(
    const float3 & p1, const float3 & q1, const float3 & r1, 
    const float3 & p2, const float3 & q2, const float3 & r2, float eps) {
    eps = 1e-6;
    // 삼각형 1과 2의 법선 계산
    float3 v1 = p2 - r2;
    float3 v2 = q2 - r2;
    float3 N2 = cross(v1, v2); // 삼각형 2의 법선

    v1 = q1 - p1;
    v2 = r1 - p1;
    float3 N1 = cross(v1, v2); // 삼각형 1의 법선

    // 법선 간의 각도 계산
    float normN1 = sqrt(dot(N1, N1));
    float normN2 = sqrt(dot(N2, N2));
    float dotN1N2 = dot(N1, N2);
    float angle = acosf(dotN1N2 / (normN1 * normN2));

    // 삼각형 1의 꼭짓점이 삼각형 2의 평면과의 거리
    float dp1 = dot(p1 - r2, N2);
    float dq1 = dot(q1 - r2, N2);
    float dr1 = dot(r1 - r2, N2);

    // 삼각형 2의 꼭짓점이 삼각형 1의 평면과의 거리
    float dp2 = dot(p2 - p1, N1);
    float dq2 = dot(q2 - p1, N1);
    float dr2 = dot(r2 - p1, N1);

    //printf("dp1 %f dq1 %f dr1 %f dp2 %f dq2 %f dr2 %f angle %f\n", dp1, dq1, dr1, dp2, dq2, dr2, angle);

    // 모든 꼭짓점이 다른 삼각형의 평면에 충분히 가까운지와 법선 각도가 충분히 작은지 확인
    return fabs(dp1) <= eps && fabs(dq1) <= eps && fabs(dr1) <= eps &&
           fabs(dp2) <= eps && fabs(dq2) <= eps && fabs(dr2) <= eps &&
           angle <= eps;
}

__device__ __host__
bool tri_tri_overlap_test_coplanar(
  const float2 &p1, const float2 &q1, const float2 &r1,
  const float2 &p2, const float2 &q2, const float2 &r2, bool &shared_vertex) 
{
  if ( ORIENT_2D(p1,q1,r1) < 0.0)
    if ( ORIENT_2D(p2,q2,r2) < 0.0)
      return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,r2,q2);
    else
      return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,q2,r2);
  else
    if ( ORIENT_2D(p2,q2,r2) < 0.0 )
      return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,r2,q2);
    else
      return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,q2,r2);
};


// ******************** coplanar detection
__device__ bool are_points_same_2d(float2 p1, float2 p2, float epsilon) {
    // return (fabs(p1.x - p2.x) < epsilon) && 
    //        (fabs(p1.y - p2.y) < epsilon);
    return (p1.x == p2.x) && (p1.y == p2.y);
}

__device__ bool is_on_same_side_2d(float2 p1, float2 p2, float2 a, float2 b) {
    float cp1 = (b.x - a.x) * (p1.y - a.y) - (b.y - a.y) * (p1.x - a.x);
    float cp2 = (b.x - a.x) * (p2.y - a.y) - (b.y - a.y) * (p2.x - a.x);
    // printf("p1: (%f, %f)\n", p1.x, p1.y);
    // printf("p2: (%f, %f)\n", p2.x, p2.y);
    // printf("a: (%f, %f)\n", a.x, a.y);
    // printf("b: (%f, %f)\n", b.x, b.y);
    // printf("cp1 cp2 %f %f\n", cp1, cp2);
    return cp1 * cp2 > 0;
}

__device__ bool check_shared_edge_and_side_2d(
    float2 p1, float2 q1, float2 r1, 
    float2 p2, float2 q2, float2 r2, 
    float epsilon) 
{
    epsilon = 1e-6;
    // Check for shared edge between the triangles
    if (are_points_same_2d(p1, p2, epsilon) && are_points_same_2d(q1, q2, epsilon)) {
        // Shared edge found between p1-q1 and p2-q2
        return is_on_same_side_2d(r1, r2, p1, q1);
    }
    if (are_points_same_2d(p1, q2, epsilon) && are_points_same_2d(q1, p2, epsilon)) {
        // Shared edge found between p1-q1 and q2-p2
        return is_on_same_side_2d(r1, r2, p1, q1);
    }
    if (are_points_same_2d(p1, p2, epsilon) && are_points_same_2d(r1, r2, epsilon)) {
        // Shared edge found between p1-r1 and p2-r2
        return is_on_same_side_2d(q1, q2, p1, r1);
    }
    if (are_points_same_2d(p1, r2, epsilon) && are_points_same_2d(r1, p2, epsilon)) {
        // Shared edge found between p1-r1 and r2-p2
        return is_on_same_side_2d(q1, q2, p1, r1);
    }
    if (are_points_same_2d(q1, q2, epsilon) && are_points_same_2d(r1, r2, epsilon)) {
        // Shared edge found between q1-r1 and q2-r2
        return is_on_same_side_2d(p1, p2, q1, r1);
    }
    if (are_points_same_2d(q1, r2, epsilon) && are_points_same_2d(r1, q2, epsilon)) {
        // Shared edge found between q1-r1 and r2-q2
        return is_on_same_side_2d(p1, p2, q1, r1);
    }

    return false; // No shared edge found, or vertices are not on the same side
}

// return : projection (P1, Q1, R1, P2, Q2, R2)
__device__
void projection_to_2D(const float3 &p1, const float3 &q1, const float3 &r1,
    const float3 &p2, const float3 &q2, const float3 &r2,
    float2 &P1, float2 &Q1, float2 &R1, float2 &P2, float2 &Q2, float2 &R2)
    {
    float3 v1, v2;//, v;
    float3 normal_1;
    v1=q1-p1;
    v2=r1-p1;
      //N1=v1.cross(v2);
    normal_1 = cross(v1, v2);

    float n_x = fabs(normal_1.x);
    float n_y = fabs(normal_1.y);
    float n_z = fabs(normal_1.z);

    /* Projection of the triangles in 3D onto 2D such that the area of
    the projection is maximized. */

    if (n_x > n_z && n_x >= n_y) {
        // Project onto plane YZ
        P1 = make_float2(q1.z, q1.y);
        Q1 = make_float2(p1.z, p1.y);
        R1 = make_float2(r1.z, r1.y);

        P2 = make_float2(q2.z, q2.y);
        Q2 = make_float2(p2.z, p2.y);
        R2 = make_float2(r2.z, r2.y);
    } else if (n_y > n_z && n_y >= n_x) {
        // Project onto plane XZ
        P1 = make_float2(q1.x, q1.z);
        Q1 = make_float2(p1.x, p1.z);
        R1 = make_float2(r1.x, r1.z);

        P2 = make_float2(q2.x, q2.z);
        Q2 = make_float2(p2.x, p2.z);
        R2 = make_float2(r2.x, r2.z);
    } else {
        // Project onto plane XY
        P1 = make_float2(p1.x, p1.y);
        Q1 = make_float2(q1.x, q1.y);
        R1 = make_float2(r1.x, r1.y);

        P2 = make_float2(p2.x, p2.y);
        Q2 = make_float2(q2.x, q2.y);
        R2 = make_float2(r2.x, r2.y);
    }
}

__device__
bool coplanar_same_side_test(
    const float3 &p1, const float3 &q1, const float3 &r1,
    const float3 &p2, const float3 &q2, const float3 &r2, float epsilon)
{
    float2 P1, Q1, R1;
    float2 P2, Q2, R2;

    // float3 v1, v2;//, v;
    // float3 normal_1;
    // v1=q1-p1;
    // v2=r1-p1;
    //   //N1=v1.cross(v2);
    // normal_1 = cross(v1, v2);

    // float n_x = fabs(normal_1.x);
    // float n_y = fabs(normal_1.y);
    // float n_z = fabs(normal_1.z);

    // /* Projection of the triangles in 3D onto 2D such that the area of
    // the projection is maximized. */

    // if (n_x > n_z && n_x >= n_y) {
    //     // Project onto plane YZ
    //     P1 = make_float2(q1.z, q1.y);
    //     Q1 = make_float2(p1.z, p1.y);
    //     R1 = make_float2(r1.z, r1.y);

    //     P2 = make_float2(q2.z, q2.y);
    //     Q2 = make_float2(p2.z, p2.y);
    //     R2 = make_float2(r2.z, r2.y);
    // } else if (n_y > n_z && n_y >= n_x) {
    //     // Project onto plane XZ
    //     P1 = make_float2(q1.x, q1.z);
    //     Q1 = make_float2(p1.x, p1.z);
    //     R1 = make_float2(r1.x, r1.z);

    //     P2 = make_float2(q2.x, q2.z);
    //     Q2 = make_float2(p2.x, p2.z);
    //     R2 = make_float2(r2.x, r2.z);
    // } else {
    //     // Project onto plane XY
    //     P1 = make_float2(p1.x, p1.y);
    //     Q1 = make_float2(q1.x, q1.y);
    //     R1 = make_float2(r1.x, r1.y);

    //     P2 = make_float2(p2.x, p2.y);
    //     Q2 = make_float2(q2.x, q2.y);
    //     R2 = make_float2(r2.x, r2.y);
    // }
    projection_to_2D(p1, q1, r1, p2, q2, r2, P1, Q1, R1, P2, Q2, R2);
    return check_shared_edge_and_side_2d(P1, Q1, R1, P2, Q2, R2, epsilon);
    //return true;
    //return tri_tri_overlap_test_coplanar(P1, Q1, R1, P2, Q2, R2, shared_vertex);  // Make sure this function is also adapted for float2.
}


__device__ bool are_points_same(float3 p1, float3 p2, float epsilon) {
    // A simple epsilon-based comparison can be used for floating point precision
    return (fabs(p1.x - p2.x) < epsilon) && 
           (fabs(p1.y - p2.y) < epsilon) && 
           (fabs(p1.z - p2.z) < epsilon);
}

__device__ float3 cross_product(float3 v1, float3 v2) {
    return make_float3(v1.y * v2.z - v1.z * v2.y,
                       v1.z * v2.x - v1.x * v2.z,
                       v1.x * v2.y - v1.y * v2.x);
}

__device__ float cross_product_2d(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

__device__ float dot_product(float3 v1, float3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ float dot_product_2d(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float2 vector_from_points_2D(float2 a, float2 b) {
    return make_float2(b.x - a.x, b.y - a.y);
}


__device__ bool is_on_same_side(float3 p1, float3 p2, float3 a, float3 b) {
    float3 cp1 = cross_product(b - a, p1 - a);
    float3 cp2 = cross_product(b - a, p2 - a);
    return dot_product(cp1, cp2) >= 0;
}

__device__ bool check_shared_edge_and_side(float3 p1, float3 q1, float3 r1, float3 p2, float3 q2, float3 r2, float epsilon) {
    // Check for shared edge between the triangles
    // Compare each edge of the first triangle with each edge of the second triangle
    if (are_points_same(p1, p2, epsilon) && are_points_same(q1, q2, epsilon)) {
        // Shared edge found, now check if the other vertices (r1 and r2) are on the same side
        return is_on_same_side(r1, r2, p1, q1);
    }
    if (are_points_same(p1, q2, epsilon) && are_points_same(q1, p2, epsilon)) {
        return is_on_same_side(r1, r2, p1, q1);
    }
    if (are_points_same(p1, p2, epsilon) && are_points_same(r1, r2, epsilon)) {
        return is_on_same_side(q1, q2, p1, r1);
    }
    if (are_points_same(p1, r2, epsilon) && are_points_same(r1, p2, epsilon)) {
        return is_on_same_side(q1, q2, p1, r1);
    }
    if (are_points_same(q1, q2, epsilon) && are_points_same(r1, r2, epsilon)) {
        return is_on_same_side(p1, p2, q1, r1);
    }
    if (are_points_same(q1, r2, epsilon) && are_points_same(r1, q2, epsilon)) {
        return is_on_same_side(p1, p2, q1, r1);
    }

    return false; // No shared edge found, or vertices are not on the same side
}

//test for vertex sharing-------------
__device__ float angle_between_vectors_2d(float2 a, float2 b) {
    float dot_prod = dot_product_2d(a, b);
    float cross_prod = cross_product_2d(a, b);
    return atan2f(cross_prod, dot_prod);
}

__device__ bool angle_range_overlap(float angle_std, float angle1, float angle2, float epsilon)
{
    epsilon = 1e-6;
    float lower_bound, upper_bound;
    bool is_angle_in_range = false;
    // Check if either angle1 or angle2 is within the range defined by angle_std
    if(angle_std < 0){
      lower_bound = angle_std - epsilon;
      is_angle_in_range = ((lower_bound <= angle1 && angle1 <= 0) || (lower_bound <= angle2 && angle2 <= 0));
      //printf("lower_bound %f is_angle_in_range %d", lower_bound, is_angle_in_range);
    }
    else{
      upper_bound = angle_std + epsilon;
      is_angle_in_range = ((0 <= angle1 && angle1 <= upper_bound) || (0 <= angle2 && angle2 <= upper_bound));
      //printf("upper_bound %f is_angle_in_range %d", upper_bound, is_angle_in_range);
    }

    // if two angle is not in range of angle_std
    // check sign and the abs value of angle (check if it is other side)
    if (!is_angle_in_range) {
        bool is_opposite_sign = (angle1 * angle2 < 0.0f);
        bool is_sum_less_than_pi = (fabs(angle1) + fabs(angle2) < M_PI);

        //printf("angle_std %f, angle1 %f, angle2 %f is_angel_in_range %d, is_opposite_sign %d, is_sum_less_than_pi %d\n", angle_std, angle1, angle2, is_angle_in_range, is_opposite_sign, is_sum_less_than_pi);
        return is_opposite_sign && is_sum_less_than_pi;
    }
    //printf("angle_std %f, angle1 %f, angle2 %f is_angel_in_range %d\n", angle_std, angle1, angle2, is_angle_in_range);
    return is_angle_in_range;
}

__device__ bool are_float2_equal(float2 a, float2 b) {
    return (a.x == b.x && a.y == b.y);
}

__device__ bool coplanar_vertex_sharing_test(float3 p1, float3 q1, float3 r1, float3 p2, float3 q2, float3 r2, float epsilon) {
    float2 P1, Q1, R1, P2, Q2, R2;
    
    projection_to_2D(p1, q1, r1, p2, q2, r2, P1, Q1, R1, P2, Q2, R2);
    float2 vec_std, vec_pair, vec_1, vec_2;
    float angle_std, angle_1, angle_2;

    // shared vertex is p1, p2
    if(are_float2_equal(P1, P2)){
        vec_std = vector_from_points_2D(P1, Q1);  // p1-q1
        vec_pair = vector_from_points_2D(P1, R1); // p1-r1
        vec_1 = vector_from_points_2D(P1, Q2);    // p1-q2
        vec_2 = vector_from_points_2D(P1, R2);    // p1-r2
    }
    // shared vertex is p1, q2
    else if(are_float2_equal(P1, Q2)){
        vec_std = vector_from_points_2D(P1, Q1);
        vec_pair = vector_from_points_2D(P1, R1);
        vec_1 = vector_from_points_2D(P1, P2);
        vec_2 = vector_from_points_2D(P1, R2);
    }
    // shared vertex is p1, r2
    else if(are_float2_equal(P1, R2)){
        vec_std = vector_from_points_2D(P1, Q1);
        vec_pair = vector_from_points_2D(P1, R1);
        vec_1 = vector_from_points_2D(P1, P2);
        vec_2 = vector_from_points_2D(P1, Q2);
    }
    // shared vertex is q1, p2
    else if(are_float2_equal(Q1, P2)){
        vec_std = vector_from_points_2D(Q1, P1);
        vec_pair = vector_from_points_2D(Q1, R1);
        vec_1 = vector_from_points_2D(Q1, Q2);
        vec_2 = vector_from_points_2D(Q1, R2);
    }
    // shared vertex is q1, q2
    else if(are_float2_equal(Q1, Q2)){
        vec_std = vector_from_points_2D(Q1, P1);
        vec_pair = vector_from_points_2D(Q1, R1);
        vec_1 = vector_from_points_2D(Q1, P2);
        vec_2 = vector_from_points_2D(Q1, R2);
    }
    // shared vertex is q1, r2
    else if(are_float2_equal(Q1, R2)){
        vec_std = vector_from_points_2D(Q1, P1);
        vec_pair = vector_from_points_2D(Q1, R1);
        vec_1 = vector_from_points_2D(Q1, P2);
        vec_2 = vector_from_points_2D(Q1, Q2);
    }
    // shared vertex is r1, p2
    else if(are_float2_equal(R1, P2)){
        vec_std = vector_from_points_2D(R1, P1);
        vec_pair = vector_from_points_2D(R1, Q1);
        vec_1 = vector_from_points_2D(R1, Q2);
        vec_2 = vector_from_points_2D(R1, R2);
    }
    // shared vertex is r1, q2
    else if(are_float2_equal(R1, Q2)){
        vec_std = vector_from_points_2D(R1, P1);
        vec_pair = vector_from_points_2D(R1, Q1);
        vec_1 = vector_from_points_2D(R1, P2);
        vec_2 = vector_from_points_2D(R1, R2);
    }
    // shared vertex is r1, r2
    else if(are_float2_equal(R1, R2)){
        vec_std = vector_from_points_2D(R1, P1);
        vec_pair = vector_from_points_2D(R1, Q1);
        vec_1 = vector_from_points_2D(R1, P2);
        vec_2 = vector_from_points_2D(R1, Q2);
    }
    else {
      // no shared vertex, return
        return false;
    }

    angle_std = angle_between_vectors_2d(vec_std, vec_pair);
    angle_1 = angle_between_vectors_2d(vec_std, vec_1);
    angle_2 = angle_between_vectors_2d(vec_std, vec_2);

    return angle_range_overlap(angle_std, angle_1, angle_2, epsilon);
    //TODO: 다른 각을 std로 잡아서 test
}

// No sharing case (using libigl to detect touch) -----------
__device__
bool coplanar_without_sharing_test(
    const float3 &p1, const float3 &q1, const float3 &r1,
    const float3 &p2, const float3 &q2, const float3 &r2)
{
    float2 P1, Q1, R1;
    float2 P2, Q2, R2;

    projection_to_2D(p1, q1, r1, p2, q2, r2, P1, Q1, R1, P2, Q2, R2);

    return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);  // Make sure this function is also adapted for float2.
}

// __device__ __host__ bool coplanar_intersection_test(
//     const float3 &p1, const float3 &q1, const float3 &r1,
//     const float3 &p2, const float3 &q2, const float3 &r2)
//   {
//     float3 N
//     v1=q1-p1;
//     v2=r1-p1;
//     N1 = cross(v1, v2);
//     if(check_if_shared_vertex(p1, q1, r1, p2, q2, r2)){

//     }
//   }
// // coplanar test
// __device__ __host__
// bool INTERSECTION_TEST_EDGE(
//   const float2 & P1, const float2 & Q1, const float2 & R1,  
//   const float2 & P2, const float2 & Q2, const float2 & R2
// )
// {
//   if (ORIENT_2D(R2,P2,Q1) >= 0.0) {
//     if (ORIENT_2D(P1,P2,Q1) >= 0.0) {
//         if (ORIENT_2D(P1,Q1,R2) >= 0.0) return true;
//         else return false;} else { 
//       if (ORIENT_2D(Q1,R1,P2) >= 0.0){ 
//   if (ORIENT_2D(R1,P1,P2) >= 0.0) return true; else return false;} 
//       else return false; } 
//   } else {
//     if (ORIENT_2D(R2,P2,R1) >= 0.0) {
//       if (ORIENT_2D(P1,P2,R1) >= 0.0) {
//   if (ORIENT_2D(P1,R1,R2) >= 0.0) return true;  
//   else {
//     if (ORIENT_2D(Q1,R1,R2) >= 0.0) return true; else return false;}}
//       else  return false; }
//     else return false; }
// }

// __device__ __host__
// bool INTERSECTION_TEST_VERTEX(
//   const float2 & P1, const float2 & Q1, const float2 & R1,  
//   const float2 & P2, const float2 & Q2, const float2 & R2
// )
// {
//   if (ORIENT_2D(R2,P2,Q1) >= 0.0)
//     if (ORIENT_2D(R2,Q2,Q1) <= 0.0)
//       if (ORIENT_2D(P1,P2,Q1) > 0.0) {
//         if (ORIENT_2D(P1,Q2,Q1) <= 0.0) return true;
//         else return false;} 
//       else {
//         if (ORIENT_2D(P1,P2,R1) >= 0.0)
//           if (ORIENT_2D(Q1,R1,P2) >= 0.0) return true; 
//           else return false;
//         else return false;
//       }
//       else 
//         if (ORIENT_2D(P1,Q2,Q1) <= 0.0)
//           if (ORIENT_2D(R2,Q2,R1) <= 0.0)
//             if (ORIENT_2D(Q1,R1,Q2) >= 0.0) return true; 
//             else return false;
//           else return false;
//         else return false;
//       else
//         if (ORIENT_2D(R2,P2,R1) >= 0.0) 
//           if (ORIENT_2D(Q1,R1,R2) >= 0.0)
//             if (ORIENT_2D(P1,P2,R1) >= 0.0) return true;
//             else return false;
//           else 
//             if (ORIENT_2D(Q1,R1,Q2) >= 0.0) {
//               if (ORIENT_2D(R2,R1,Q2) >= 0.0) return true; 
//               else return false; 
//             }
//         else return false; 
//   else  return false; 
// }

// __device__ __host__
// bool ccw_tri_tri_intersection_2d(
//   const float2 &p1, const float2 &q1, const float2 &r1,
//   const float2 &p2, const float2 &q2, const float2 &r2)
//   {
//   if ( ORIENT_2D(p2,q2,p1) >= 0.0 ) {
//     if ( ORIENT_2D(q2,r2,p1) >= 0.0 ) {
//       if ( ORIENT_2D(r2,p2,p1) >= 0.0 ) return true;
//       else return INTERSECTION_TEST_EDGE(p1,q1,r1,p2,q2,r2);
//     } else {  
//       if ( ORIENT_2D(r2,p2,p1) >= 0.0 ) 
//       return INTERSECTION_TEST_EDGE(p1,q1,r1,r2,p2,q2);
//       else return INTERSECTION_TEST_VERTEX(p1,q1,r1,p2,q2,r2);}}
//   else {
//     if ( ORIENT_2D(q2,r2,p1) >= 0.0 ) {
//       if ( ORIENT_2D(r2,p2,p1) >= 0.0 ) 
//         return INTERSECTION_TEST_EDGE(p1,q1,r1,q2,r2,p2);
//       else  return INTERSECTION_TEST_VERTEX(p1,q1,r1,q2,r2,p2);}
//     else return INTERSECTION_TEST_VERTEX(p1,q1,r1,r2,p2,q2);}
// };

// __device__ __host__
// bool tri_tri_overlap_test_2d(
//   const float2 &p1, const float2 &q1, const float2 &r1,
//   const float2 &p2, const float2 &q2, const float2 &r2) 
// {
//   if ( ORIENT_2D(p1,q1,r1) < 0.0)
//     if ( ORIENT_2D(p2,q2,r2) < 0.0)
//       return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,r2,q2);
//     else
//       return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,q2,r2);
//   else
//     if ( ORIENT_2D(p2,q2,r2) < 0.0 )
//       return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,r2,q2);
//     else
//       return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,q2,r2);
// };



// __device__ __host__
// bool coplanar_tri_tri3d(
//     const float3 &p1, const float3 &q1, const float3 &r1,
//     const float3 &p2, const float3 &q2, const float3 &r2,
//     const float3 &normal_1)
// {
//     float2 P1, Q1, R1;
//     float2 P2, Q2, R2;

//     float n_x = fabs(normal_1.x);
//     float n_y = fabs(normal_1.y);
//     float n_z = fabs(normal_1.z);

//     /* Projection of the triangles in 3D onto 2D such that the area of
//     the projection is maximized. */

//     if (n_x > n_z && n_x >= n_y) {
//         // Project onto plane YZ
//         P1 = make_float2(q1.z, q1.y);
//         Q1 = make_float2(p1.z, p1.y);
//         R1 = make_float2(r1.z, r1.y);

//         P2 = make_float2(q2.z, q2.y);
//         Q2 = make_float2(p2.z, p2.y);
//         R2 = make_float2(r2.z, r2.y);
//     } else if (n_y > n_z && n_y >= n_x) {
//         // Project onto plane XZ
//         P1 = make_float2(q1.x, q1.z);
//         Q1 = make_float2(p1.x, p1.z);
//         R1 = make_float2(r1.x, r1.z);

//         P2 = make_float2(q2.x, q2.z);
//         Q2 = make_float2(p2.x, p2.z);
//         R2 = make_float2(r2.x, r2.z);
//     } else {
//         // Project onto plane XY
//         P1 = make_float2(p1.x, p1.y);
//         Q1 = make_float2(q1.x, q1.y);
//         R1 = make_float2(r1.x, r1.y);

//         P2 = make_float2(p2.x, p2.y);
//         Q2 = make_float2(q2.x, q2.y);
//         R2 = make_float2(r2.x, r2.y);
//     }

//     return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);  // Make sure this function is also adapted for float2.
// }

// detect point intersection (점끼리 겹쳐도 intersect로 판정)
// TODO: 리스트에 인접한 삼각형을 제거해야 함
// bool tri_tri_intersect(Triangle<float3>* d_triangles_raw, unsigned int* query_list_raw, unsigned int* num_found_list_raw, int N, unsigned int buffer_size, cusimp::Triangle<int> *F, cusimp::CUSimp *sp){
//     // ... 생략 ...
//     //     // 호스트 메모리에 대한 임시 저장소 생성
//     // cusimp::Triangle<int> *host_F = new cusimp::Triangle<int>[N];

//     // // 디바이스에서 호스트로 메모리 복사
//     // cudaMemcpy(host_F, F, N * sizeof(cusimp::Triangle<int>), cudaMemcpyDeviceToHost);

//     // // F 출력
//     // std::cout << "F values:" << std::endl;
//     // for (int i = 0; i < N; ++i) {
//     //     std::cout << "Triangle " << i << ": (" 
//     //               << host_F[i].i << ", " 
//     //               << host_F[i].j << ", " 
//     //               << host_F[i].k << ")" << std::endl;
//     // }

//     // // 호스트 메모리 해제
//     // delete[] host_F;
//     /*
//     unsigned int buffer[BUFFER_SIZE];
//     unsigned int num_found;
//     bool coplanar;
//     //unsigned int num_found_query = num_found_query_list_raw[idx];
//     Eigen::RowVector3d source, target;
//     // query triangle
//     Eigen::RowVector3d p1 = float3ToRowVector3d(triangles[4].v0);
//     Eigen::RowVector3d q1 = float3ToRowVector3d(triangles[4].v1);
//     Eigen::RowVector3d r1 = float3ToRowVector3d(triangles[4].v2);

//     // list triangle
//     Eigen::RowVector3d p2 = float3ToRowVector3d(triangles[11].v0);
//     Eigen::RowVector3d q2 = float3ToRowVector3d(triangles[11].v1);
//     Eigen::RowVector3d r2 = float3ToRowVector3d(triangles[11].v2);                            
//     bool isIntersecting = igl::tri_tri_intersection_test_3d(p1, q1, r1, p2, q2, r2, coplanar, source, target);
//     return isIntersecting;
//     */

//     //thrust::device_vector<Triangle> d_triangles = triangles;
//     //Triangle* d_triangles_raw = thrust::raw_pointer_cast(d_triangles.data());

//     unsigned int h_isIntersect = 0; // host에서 사용할 변수
//     unsigned int* d_isIntersect;        // device에서 사용할 포인터
//     cudaMalloc((void**)&d_isIntersect, sizeof(unsigned int));
//     cudaMemcpy(d_isIntersect, &h_isIntersect, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
//     // check where is intersection -------------------
    
//     const int maxIntersections = N;
//     unsigned int* d_intersections;
//     cudaMalloc((void**)&d_intersections, 2 * maxIntersections * sizeof(unsigned int));
//     unsigned int h_val = 0;
//     cudaMemcpy(d_intersections + 2 * maxIntersections - 2, &h_val, sizeof(unsigned int), cudaMemcpyHostToDevice); // Counter를 초기화
//     //cudaMalloc((void**)&sp->intersected_triangle_idx, 2 * maxIntersections * sizeof(unsigned int));
//     //-------------------------
//     //std::vector<unsigned int> h_buffer(N * buffer_size);  // 호스트에 저장할 버퍼 (프린트를 위함)
//     //thrust::device_vector<unsigned int> dev_buffer(N * buffer_size);
//     //unsigned int* dev_buffer_raw = thrust::raw_pointer_cast(dev_buffer.data());
//     // d_intersection : where is the intersection occur?

//     // invalid 한 것을 리턴 하는가?
//     // 1. Device에 대한 변수 추가
//     unsigned int* d_invalid_indices;
//     unsigned int* d_invalid_count;
//     cudaMalloc(&d_invalid_count, sizeof(unsigned int));
//     // 최대 가능한 잘못된 인덱스 수에 대한 메모리 할당 (N은 전체 개수)
//     cudaMalloc(&d_invalid_indices, N * sizeof(unsigned int));
    
//     unsigned int h_invalid_count = 0;
//     cudaMemcpy(d_invalid_count, &h_invalid_count, sizeof(unsigned int), cudaMemcpyHostToDevice);

//     unsigned int* d_pos;
//     cudaMalloc(&d_pos, sizeof(unsigned int));
//     cudaMemset(d_pos, 0, sizeof(unsigned int));
    

//     // TODO: cuda mem error here
//     thrust::for_each(thrust::device,
//                     thrust::make_counting_iterator<std::size_t>(0),//0
//                     thrust::make_counting_iterator<std::size_t>(N),//N
//                     [d_pos, F, query_list_raw, num_found_list_raw, d_triangles_raw, d_isIntersect, buffer_size, d_intersections, maxIntersections, d_invalid_indices, d_invalid_count] __device__(std::size_t idx) {
//                     //[F, query_list_raw, num_found_list_raw, d_triangles_raw, d_isIntersect, buffer_size, d_intersections, maxIntersections] __device__(std::size_t idx) {
//                     //[query_list_raw, num_found_list_raw, d_triangles_raw, d_isIntersect, buffer_size] __device__(std::size_t idx){
//                         // invalid triangle
//                         if(F[idx].i == -1) {
//                           // 잘못된 인덱스 저장
//                           unsigned int pos = atomicAdd(d_invalid_count, 1);
//                           d_invalid_indices[pos] = idx;
//                           return;
//                         }
//                         bool coplanar;
//                         unsigned int num_found = num_found_list_raw[idx];
//                         float3 source, target;
//                         float3 p1 = d_triangles_raw[idx].v0;
//                         float3 q1 = d_triangles_raw[idx].v1;
//                         float3 r1 = d_triangles_raw[idx].v2;
//                         for(unsigned int i = 0; i < num_found; i++){
//                             unsigned int idx_buffer = query_list_raw[idx * buffer_size + i];
//                             if(idx_buffer == 0xFFFFFFFF) continue;
//                             if(F[idx_buffer].i == -1) continue; // invalid triangle
//                             //dev_buffer_raw[idx*buffer_size+i] = idx_buffer;
//                             //printf("queryidx : %d, comp index : %d\n", idx, query_list_raw[idx * buffer_size + i]);
//                             float3 p2 = d_triangles_raw[idx_buffer].v0;
//                             float3 q2 = d_triangles_raw[idx_buffer].v1;
//                             float3 r2 = d_triangles_raw[idx_buffer].v2;                            
//                             bool isIntersecting = tri_tri_intersection_test_3d(p1, q1, r1, p2, q2, r2, coplanar, source, target);
//                             if(isIntersecting) {
//                                 atomicExch(d_isIntersect, 1);
//                                 // check where is intersection -------------------
                                
//                                 int pos = atomicAdd(d_pos, 2);
//                                 if (pos < 2 * maxIntersections - 2) {
//                                     d_intersections[pos] = idx;
//                                     d_intersections[pos + 1] = idx_buffer;
//                                     //d_intersections[pos + 2] = coplanar ? 1 : 0;  // coplanar 정보 저장
//                                 }
//                             }
//                           }
//                         return;
//                     });
// //cudaMalloc((void**)&sp->intersected_triangle_idx, 2 * maxIntersections * sizeof(unsigned int));                    
// cudaMemcpy(sp->intersected_triangle_idx, d_intersections, 2 * maxIntersections * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
// //cudaMalloc((void**)&sp->n_intersect, sizeof(unsigned int));
// cudaMemcpy(sp->n_intersect, d_pos, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
// cudaFree(d_intersections);
// cudaFree(d_pos);

// // 지금 invalid triangle 을 return 하는가?
// // unsigned int* h_invalid_indices = new unsigned int[N];
// // cudaMemcpy(h_invalid_indices, d_invalid_indices, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
// // cudaMemcpy(&h_invalid_count, d_invalid_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

// // std::cout << "Invalid indices: ";
// // for (unsigned int i = 0; i < h_invalid_count; ++i) {
// //     std::cout << h_invalid_indices[i] << " ";
// // }
// // std::cout << std::endl;

// // 메모리 해제
// // delete[] h_invalid_indices;
// // cudaFree(d_invalid_indices);
// // cudaFree(d_invalid_count);

//     /*
//     thrust::copy(dev_buffer.begin(), dev_buffer.end(), h_buffer.begin());
//     for(int i = 0; i < N; ++i) {
//         for(int j = 0; j < buffer_size; ++j) {
//             std::cout << "buffer value at " << j << " for query " << i << ": " << h_buffer[i * buffer_size + j] << std::endl;
//         }
//     }
// */
//   //print where is intersection ----------
  
  
//     // unsigned int* h_intersections = new unsigned int[2 * maxIntersections];
//     // cudaMemcpy(h_intersections, d_intersections, 2 * maxIntersections * sizeof(unsigned int), cudaMemcpyDeviceToHost);

//     // for (int i = 0; i < h_intersections[2 * maxIntersections - 2]; i += 2) {    
//     //     printf("Intersection between triangle %d and triangle %d.\n", 
//     //         h_intersections[i], h_intersections[i + 1]);
//     // }

//     // unsigned int h_pos;
//     // cudaMemcpy(&h_pos, d_pos, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     // printf("# of intersection : %d\n", h_pos);
    
//   // ----------------
//     cudaDeviceSynchronize();
//     cudaMemcpy(&h_isIntersect, d_isIntersect, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaFree(d_isIntersect);

//     // free memory used at intersection check 
//     cudaFree(d_intersections);

//     // free memory used at print intersection
//     //delete[] h_intersections;

//     return h_isIntersect == 1;
// /*
//     cudaMemcpy(&h_isIntersect, d_isIntersect, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaFree(d_isIntersect);

//     return h_isIntersect == 1;
//     */
// }
}
#endif