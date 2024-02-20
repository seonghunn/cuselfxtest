#ifndef LBVH_TRI_TRI_INTERSECT_CUH
#define LBVH_TRI_TRI_INTERSECT_CUH
#include "types.cuh"
#include "math.cuh"
#include <thrust/for_each.h>
#include <vector>
#include <cmath>

namespace selfx{
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

  __device__ 
  inline bool are_points_same(float3 p1, float3 p2, float epsilon) {
      // A simple epsilon-based comparison can be used for floating point precision
      return (fabs(p1.x - p2.x) < epsilon) && 
             (fabs(p1.y - p2.y) < epsilon) && 
             (fabs(p1.z - p2.z) < epsilon);
  }

  __device__ 
  inline float3 cross_product(float3 v1, float3 v2) {
      return make_float3(v1.y * v2.z - v1.z * v2.y,
                         v1.z * v2.x - v1.x * v2.z,
                         v1.x * v2.y - v1.y * v2.x);
  }

  __device__ 
  inline float cross_product_2d(float2 a, float2 b) {
      return a.x * b.y - a.y * b.x;
  }

  __device__ 
  inline float dot_product(float3 v1, float3 v2) {
      return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  }

  __device__ 
  inline float dot_product_2d(float2 a, float2 b) {
      return a.x * b.x + a.y * b.y;
  }

  __device__ 
  inline float2 vector_from_points_2D(float2 a, float2 b) {
      return make_float2(b.x - a.x, b.y - a.y);
  }

  __device__ __host__
  inline float ORIENT_2D(const float2& a, const float2& b, const float2& c)
  {
      return (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);
  }
  
  __device__ 
  inline bool are_points_same_2d(float2 p1, float2 p2, float epsilon) {
      return (p1.x == p2.x) && (p1.y == p2.y);
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

  __device__ bool is_on_same_side_2d(float2 p1, float2 p2, float2 a, float2 b) {
      float cp1 = (b.x - a.x) * (p1.y - a.y) - (b.y - a.y) * (p1.x - a.x);
      float cp2 = (b.x - a.x) * (p2.y - a.y) - (b.y - a.y) * (p2.x - a.x);
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
          // shared edge p1-q1, p2-q2
          return is_on_same_side_2d(r1, r2, p1, q1);
      }

      if (are_points_same_2d(p1, q2, epsilon) && are_points_same_2d(q1, p2, epsilon)) {
          // shared edge p1-q1, p2-q2
          return is_on_same_side_2d(r1, r2, p1, q1);
      }

      if (are_points_same_2d(p1, p2, epsilon) && are_points_same_2d(q1, r2, epsilon)) {
          // shared edge p1-q1, p2-r2
          return is_on_same_side_2d(q2, r1, p1, q1);
      }

      if (are_points_same_2d(p1, r2, epsilon) && are_points_same_2d(q1, p2, epsilon)) {
          // shared edge p1-q1, p2-r2
          return is_on_same_side_2d(q2, r1, p1, q1);
      }

      if (are_points_same_2d(p1, p2, epsilon) && are_points_same_2d(r1, q2, epsilon)) {
          // shared edge p1-r1, p2-q2
          return is_on_same_side_2d(q1, r2, p1, r1);
      }

      if (are_points_same_2d(p1, q2, epsilon) && are_points_same_2d(r1, p2, epsilon)) {
          // shared edge p1-r1, p2-q2
          return is_on_same_side_2d(q1, r2, p1, r1);
      }

      if (are_points_same_2d(p1, p2, epsilon) && are_points_same_2d(r1, r2, epsilon)) {
          // shared edge p1-r1, p2-r2
          return is_on_same_side_2d(q1, q2, p1, r1);
      }

      if (are_points_same_2d(p1, r2, epsilon) && are_points_same_2d(r1, p2, epsilon)) {
          // shared edge p1-r1, p2-r2
          return is_on_same_side_2d(q1, q2, p1, r1);
      }

      if (are_points_same_2d(q1, p2, epsilon) && are_points_same_2d(r1, q2, epsilon)) {
          // shared edge q1-r1, p2-q2
          return is_on_same_side_2d(p1, r2, q1, r1);
      }

      if (are_points_same_2d(q1, q2, epsilon) && are_points_same_2d(r1, p2, epsilon)) {
          // shared edge q1-r1, p2-q2
          return is_on_same_side_2d(p1, r2, q1, r1);
      }

      if (are_points_same_2d(q1, p2, epsilon) && are_points_same_2d(r1, r2, epsilon)) {
          // shared edge q1-r1, p2-r2
          return is_on_same_side_2d(p1, q2, q1, r1);
      }

      if (are_points_same_2d(q1, r2, epsilon) && are_points_same_2d(r1, p2, epsilon)) {
          // shared edge q1-r1, p2-r2
          return is_on_same_side_2d(p1, q2, q1, r1);
      }

      if (are_points_same_2d(r1, p2, epsilon) && are_points_same_2d(q1, r2, epsilon)) {
          // shared edge r1-q1, p2-r2
          return is_on_same_side_2d(p1, q2, r1, q1);
      }

      if (are_points_same_2d(r1, r2, epsilon) && are_points_same_2d(q1, p2, epsilon)) {
          // shared edge r1-q1, p2-r2
          return is_on_same_side_2d(p1, q2, r1, q1);
      }

      if (are_points_same_2d(p1, q2, epsilon) && are_points_same_2d(q1, r2, epsilon)){
          // shared edge p1-q1, q2-r2
          return is_on_same_side_2d(p2, r1, p1, q1);
      }

      if (are_points_same_2d(p1, r2, epsilon) && are_points_same_2d(q1,q2, epsilon)){
          // shared edge p1-q1, q2-r2
          return is_on_same_side_2d(p2, r1, p1, q1);
      }

      if (are_points_same_2d(p1, q2, epsilon) && are_points_same_2d(r1, r2, epsilon)){
          // shared edge p1-r1, q2-r2
          return is_on_same_side_2d(p2, q1, p1, r1);
      }

      if (are_points_same_2d(p1, r2, epsilon) && are_points_same_2d(r1, q2, epsilon)){
          // shared edge p1-r1, q2-r2
          return is_on_same_side_2d(p2, q1, p1, r1);
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

  // edge sharing case
  __device__
  bool coplanar_same_side_test(
      const float3 &p1, const float3 &q1, const float3 &r1,
      const float3 &p2, const float3 &q2, const float3 &r2, float epsilon)
  {
      float2 P1, Q1, R1;
      float2 P2, Q2, R2;

      projection_to_2D(p1, q1, r1, p2, q2, r2, P1, Q1, R1, P2, Q2, R2);
      return check_shared_edge_and_side_2d(P1, Q1, R1, P2, Q2, R2, epsilon);
  }

  __device__ bool is_on_same_side(float3 p1, float3 p2, float3 a, float3 b) {
      float3 cp1 = cross_product(b - a, p1 - a);
      float3 cp2 = cross_product(b - a, p2 - a);
      return dot_product(cp1, cp2) >= 0;
  }

  //test for vertex sharing-------------
  __device__ float angle_between_vectors_2d(float2 a, float2 b) {
      float dot_prod = dot_product_2d(a, b);
      float cross_prod = cross_product_2d(a, b);
      return atan2f(cross_prod, dot_prod);
  }

  __device__ bool angle_range_overlap(float angle_std, float angle1, float angle2, float epsilon)
  {
      //epsilon = 1e-6;
      epsilon = 0;
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

  // angle overlapping test (shared vertex case)
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
  }

  // Not sharing any vertex (use libigl to detect touch)
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