#ifndef LBVH_TRI_TRI_INTERSECT_CUH
#define LBVH_TRI_TRI_INTERSECT_CUH
#include "types.h"
#include "math.cuh"
#include <thrust/for_each.h>
#include <vector>
#include <cmath>

namespace selfx{

__device__ __host__ inline bool operator==(const float3 &a, const float3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device__ __host__ inline bool operator!=(const float3 &a, const float3 &b) {
    return !(a == b);
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

}
#endif