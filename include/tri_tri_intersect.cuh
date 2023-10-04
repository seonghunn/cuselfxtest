#ifndef LBVH_TRI_TRI_INTERSECT_CUH
#define LBVH_TRI_TRI_INTERSECT_CUH
#include "types.h"
#include "math.cuh"
#include <thrust/for_each.h>
#include <Eigen/Dense>
#include <vector>

__device__
Eigen::RowVector3d float3ToRowVector3d(float3 v) {
    return Eigen::RowVector3d(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
}

namespace lbvh{

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
        return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1);
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
  float3 v1, v2, v;
  float3 N1, N2, N;
  float alpha;
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
        //return true;
        return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1);
      }
    }
  }
};

// detect point intersection (점끼리 겹쳐도 intersect로 판정)
// TODO: 리스트에 인접한 삼각형을 제거해야 함
bool tri_tri_intersect(Triangle* d_triangles_raw, unsigned int* query_list_raw, unsigned int* num_found_list_raw, int N, unsigned int buffer_size){
    // ... 생략 ...
    /*
    unsigned int buffer[BUFFER_SIZE];
    unsigned int num_found;
    bool coplanar;
    //unsigned int num_found_query = num_found_query_list_raw[idx];
    Eigen::RowVector3d source, target;
    // query triangle
    Eigen::RowVector3d p1 = float3ToRowVector3d(triangles[4].v0);
    Eigen::RowVector3d q1 = float3ToRowVector3d(triangles[4].v1);
    Eigen::RowVector3d r1 = float3ToRowVector3d(triangles[4].v2);

    // list triangle
    Eigen::RowVector3d p2 = float3ToRowVector3d(triangles[11].v0);
    Eigen::RowVector3d q2 = float3ToRowVector3d(triangles[11].v1);
    Eigen::RowVector3d r2 = float3ToRowVector3d(triangles[11].v2);                            
    bool isIntersecting = igl::tri_tri_intersection_test_3d(p1, q1, r1, p2, q2, r2, coplanar, source, target);
    return isIntersecting;
    */

    //thrust::device_vector<Triangle> d_triangles = triangles;
    //Triangle* d_triangles_raw = thrust::raw_pointer_cast(d_triangles.data());

    unsigned int h_isIntersect = 0; // host에서 사용할 변수
    unsigned int* d_isIntersect;        // device에서 사용할 포인터
    cudaMalloc((void**)&d_isIntersect, sizeof(unsigned int));
    cudaMemcpy(d_isIntersect, &h_isIntersect, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // check where is intersection -------------------
    
    const int maxIntersections = N * buffer_size;
    unsigned int* d_intersections;
    cudaMalloc((void**)&d_intersections, 3 * maxIntersections * sizeof(unsigned int));
    unsigned int h_val = 0;
    cudaMemcpy(d_intersections + 3 * maxIntersections - 3, &h_val, sizeof(unsigned int), cudaMemcpyHostToDevice); // Counter를 초기화

    //-------------------------
    //std::vector<unsigned int> h_buffer(N * buffer_size);  // 호스트에 저장할 버퍼 (프린트를 위함)
    //thrust::device_vector<unsigned int> dev_buffer(N * buffer_size);
    //unsigned int* dev_buffer_raw = thrust::raw_pointer_cast(dev_buffer.data());

    thrust::for_each(thrust::device,
                    thrust::make_counting_iterator<std::size_t>(0),//0
                    thrust::make_counting_iterator<std::size_t>(N),//N
                    [query_list_raw, num_found_list_raw, d_triangles_raw, d_isIntersect, buffer_size, d_intersections, maxIntersections] __device__(std::size_t idx) {
                    //[query_list_raw, num_found_list_raw, d_triangles_raw, d_isIntersect, buffer_size] __device__(std::size_t idx){
                        bool coplanar;
                        unsigned int num_found = num_found_list_raw[idx];
                        float3 source, target;
                        float3 p1 = d_triangles_raw[idx].v0;
                        float3 q1 = d_triangles_raw[idx].v1;
                        float3 r1 = d_triangles_raw[idx].v2;

                        for(unsigned int i = 0; i < num_found; i++){
                            unsigned int idx_buffer = query_list_raw[idx * buffer_size + i];
                            if(idx_buffer == 0xFFFFFFFF) return;
                            //dev_buffer_raw[idx*buffer_size+i] = idx_buffer;
                            //printf("queryidx : %d, comp index : %d\n", idx, query_list_raw[idx * buffer_size + i]);
                            float3 p2 = d_triangles_raw[idx_buffer].v0;
                            float3 q2 = d_triangles_raw[idx_buffer].v1;
                            float3 r2 = d_triangles_raw[idx_buffer].v2;                            
                            bool isIntersecting = tri_tri_intersection_test_3d(p1, q1, r1, p2, q2, r2, coplanar, source, target);
                            if(isIntersecting) {
                                atomicExch(d_isIntersect, 1);
                                // check where is intersection -------------------
                                
                                int pos = atomicAdd(&d_intersections[3 * maxIntersections - 3], 3);
                                if (pos < 3 * maxIntersections - 3) {
                                    d_intersections[pos] = idx;
                                    d_intersections[pos + 1] = idx_buffer;
                                    d_intersections[pos + 2] = coplanar ? 1 : 0;  // coplanar 정보 저장
                                }
                            }
                          }
                        return;
                    });

    /*
    thrust::copy(dev_buffer.begin(), dev_buffer.end(), h_buffer.begin());
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < buffer_size; ++j) {
            std::cout << "buffer value at " << j << " for query " << i << ": " << h_buffer[i * buffer_size + j] << std::endl;
        }
    }
*/
  //print where is intersection ----------
  
    unsigned int* h_intersections = new unsigned int[3 * maxIntersections];
    cudaMemcpy(h_intersections, d_intersections, 3 * maxIntersections * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_intersections[3 * maxIntersections - 3]; i += 3) {    
        printf("Intersection between triangle %d and triangle %d. Coplanar: %s\n", 
            h_intersections[i], h_intersections[i + 1], h_intersections[i + 2] ? "Yes" : "No");
    }
    
  // ----------------

    cudaMemcpy(&h_isIntersect, d_isIntersect, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    delete[] h_intersections;
    cudaFree(d_intersections);
    cudaFree(d_isIntersect);

    return h_isIntersect == 1;

/*
    cudaMemcpy(&h_isIntersect, d_isIntersect, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_isIntersect);

    return h_isIntersect == 1;
    */
}
}
#endif