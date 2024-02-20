#ifndef LBVH_SELF_INTERSECT_CUH
#define LBVH_SELF_INTERSECT_CUH
#include <vector>
#include <iostream>
#include "bvh.cuh"
#include "query.cuh"
#include "types.cuh"
#include "tri_tri_intersect.cuh"
#include "tri_tri_3d_blender.cuh"

// Average expected number of intersection candidates
#define BUFFER_SIZE 512
#define BLOCK_SIZE 512

using namespace std;

namespace selfx{
    __device__ __host__
    inline bool are_vertices_same(const float3 v1, const float3 v2, float epsilon){
        //float epsilon = 1e-4;
        return std::abs(v1.x - v2.x) < epsilon &&
           std::abs(v1.y - v2.y) < epsilon &&
           std::abs(v1.z - v2.z) < epsilon;
    }

    __device__ __host__
    inline bool detect_zero_face(const float3 p, const float3 q, const float3 r){
        float epsilon = 1e-4;
        bool same_pq = are_vertices_same(p, q, epsilon);
        bool same_qr = are_vertices_same(q, r, epsilon);
        bool same_rp = are_vertices_same(r, p, epsilon);
        return (same_pq && same_qr && same_rp);
    }

    // if two vertices are same in triangle
    __device__ __host__
    inline bool detect_invalid_face(const float3 p, const float3 q, const float3 r){
        float epsilon = 1e-5;
        bool same_pq = are_vertices_same(p, q, epsilon);
        bool same_qr = are_vertices_same(q, r, epsilon);
        bool same_rp = are_vertices_same(r, p, epsilon);
        return (same_pq || same_qr || same_rp);
    }

    // if triangle pair has shared edge
    __device__ __host__
    inline bool detect_shared_edge_coord(const float3 p1, const float3 q1, const float3 r1, 
                            const float3 p2, const float3 q2, const float3 r2){
        float epsilon = 1e-6;
        int shared_vertices = 0;
        shared_vertices += are_vertices_same(p1, p2, epsilon) || are_vertices_same(p1, q2, epsilon) || are_vertices_same(p1, r2, epsilon);
        shared_vertices += are_vertices_same(q1, p2, epsilon) || are_vertices_same(q1, q2, epsilon) || are_vertices_same(q1, r2, epsilon);
        shared_vertices += are_vertices_same(r1, p2, epsilon) || are_vertices_same(r1, q2, epsilon) || are_vertices_same(r1, r2, epsilon);

        return shared_vertices >= 2;
    }

    __device__ __host__
    inline bool detect_same_plane_coord(const float3 p1, const float3 q1, const float3 r1, 
                           const float3 p2, const float3 q2, const float3 r2){
        float epsilon = 1e-6;
        return (are_vertices_same(p1, p2, epsilon) && are_vertices_same(q1, q2, epsilon) && are_vertices_same(r1, r2, epsilon)) ||
               (are_vertices_same(p1, p2, epsilon) && are_vertices_same(q1, r2, epsilon) && are_vertices_same(r1, q2, epsilon)) ||
               (are_vertices_same(p1, q2, epsilon) && are_vertices_same(q1, p2, epsilon) && are_vertices_same(r1, r2, epsilon)) ||
               (are_vertices_same(p1, q2, epsilon) && are_vertices_same(q1, r2, epsilon) && are_vertices_same(r1, p2, epsilon)) ||
               (are_vertices_same(p1, r2, epsilon) && are_vertices_same(q1, p2, epsilon) && are_vertices_same(r1, q2, epsilon)) ||
               (are_vertices_same(p1, r2, epsilon) && are_vertices_same(q1, q2, epsilon) && are_vertices_same(r1, p2, epsilon));
    }

    __global__ void warmupKernel() {}

    __global__ void compute_num_of_query_result_kernel(
        Triangle<int>* F_d_raw,
        lbvh::bvh_device<float, Face<float3>> bvh_dev,
        unsigned int* num_found_query_raw,
        std::size_t num_faces)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_faces) return;
        // Check if the index is valid
        if (F_d_raw[idx].i == -1) {
            num_found_query_raw[idx] = 0;
            return;
        }


        unsigned int buffer[BUFFER_SIZE];

        // Get query object
        const auto self = bvh_dev.objects[idx];

        // Set query AABB
        lbvh::aabb<float> query_box;
        float minX = fminf(self.v0.x, fminf(self.v1.x, self.v2.x));
        float minY = fminf(self.v0.y, fminf(self.v1.y, self.v2.y));
        float minZ = fminf(self.v0.z, fminf(self.v1.z, self.v2.z));

        float maxX = fmaxf(self.v0.x, fmaxf(self.v1.x, self.v2.x));
        float maxY = fmaxf(self.v0.y, fmaxf(self.v1.y, self.v2.y));
        float maxZ = fmaxf(self.v0.z, fmaxf(self.v1.z, self.v2.z));

        query_box.lower = make_float4(minX, minY, minZ, 0);
        query_box.upper = make_float4(maxX, maxY, maxZ, 0);
        // Perform the query
        unsigned int num_found = lbvh::get_number_of_intersect_candidates(bvh_dev, lbvh::overlaps(query_box), buffer, idx);

        // Copy results to the device vector
        num_found_query_raw[idx] = 2 * num_found;
    }

    __global__ void compute_query_list_kernel(
        Triangle<int>* F_d_raw,
        lbvh::bvh_device<float, Face<float3>> bvh_dev,
        unsigned int* num_found_query_raw,
        unsigned int* first_query_result_raw,
        unsigned int* buffer_results_query_raw,
        std::size_t num_faces)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_faces) return;
        if (F_d_raw[idx].i == -1) {
            return;
        }

        // Get query object
        const auto self = bvh_dev.objects[idx];

        // Set query AABB
        lbvh::aabb<float> query_box;
        float minX = fminf(self.v0.x, fminf(self.v1.x, self.v2.x));
        float minY = fminf(self.v0.y, fminf(self.v1.y, self.v2.y));
        float minZ = fminf(self.v0.z, fminf(self.v1.z, self.v2.z));

        float maxX = fmaxf(self.v0.x, fmaxf(self.v1.x, self.v2.x));
        float maxY = fmaxf(self.v0.y, fmaxf(self.v1.y, self.v2.y));
        float maxZ = fmaxf(self.v0.z, fmaxf(self.v1.z, self.v2.z));

        query_box.lower = make_float4(minX, minY, minZ, 0);
        query_box.upper = make_float4(maxX, maxY, maxZ, 0);

        // Compute the query
        int first = first_query_result_raw[idx];
        unsigned int num_found = lbvh::query_device(bvh_dev, lbvh::overlaps(query_box), buffer_results_query_raw, idx, first);
    }

    bool self_intersect(Vertex<float> *V, Triangle<int> *F, unsigned int num_vertices, unsigned int num_faces) {
    float epsilon = 5e-3;
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // get triangle data to build bvh -----------------
    cudaEvent_t start_init, stop_init;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventRecord(start_init);

    // memcpy to device vector(GPU)
    thrust::device_vector<Vertex<float>> V_d(V, V + num_vertices);
    thrust::device_vector<Triangle<int>> F_d(F, F + num_faces);
    thrust::device_vector<Face<float3>> triangles_d(num_faces);

    Vertex<float>* V_d_raw = thrust::raw_pointer_cast(V_d.data());
    Triangle<int>* F_d_raw = thrust::raw_pointer_cast(F_d.data());
    Face<float3>* triangles_d_raw = thrust::raw_pointer_cast(triangles_d.data());

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator<std::size_t>(num_faces),
                     [V_d_raw, F_d_raw, triangles_d_raw] __device__(std::size_t idx){
                        Face<float3> tri;
                        int v0_row = F_d_raw[idx].i;
                        int v1_row = F_d_raw[idx].j;
                        int v2_row = F_d_raw[idx].k;
                        tri.v0 = make_float3(V_d_raw[v0_row].x, V_d_raw[v0_row].y, V_d_raw[v0_row].z);
                        tri.v1 = make_float3(V_d_raw[v1_row].x, V_d_raw[v1_row].y, V_d_raw[v1_row].z);
                        tri.v2 = make_float3(V_d_raw[v2_row].x, V_d_raw[v2_row].y, V_d_raw[v2_row].z);
                        triangles_d_raw[idx] = tri;

                        return;
                     });

    cudaEventRecord(stop_init);
    cudaEventSynchronize(stop_init);
    float milliseconds_init = 0;
    cudaEventElapsedTime(&milliseconds_init, start_init, stop_init);
    printf("triangle init runtime %f ms\n", milliseconds_init);
    cudaEventDestroy(start_init);
    cudaEventDestroy(stop_init);

    //construct bvh -------------------------------
    cudaEvent_t start0, stop0;
    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);
    cudaEventRecord(start0);

    lbvh::bvh<float, Face<float3>, aabb_getter> bvh(triangles_d.begin(), triangles_d.end(), false);
    // get device ptr
    const auto bvh_dev = bvh.get_device_repr();
    
    cudaEventRecord(stop0);
    cudaEventSynchronize(stop0);
    float milliseconds0 = 0;
    cudaEventElapsedTime(&milliseconds0, start0, stop0);
    printf("bvh construction runtime %f ms\n", milliseconds0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    // compute query list ----------------------------------
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    thrust::device_vector<unsigned int> num_found_results_dev(num_faces);
    thrust::device_vector<unsigned int> buffer_results_dev(num_faces * BUFFER_SIZE, 0xFFFFFFFF);
    thrust::device_vector<unsigned int> first_query_result(num_faces + 1);

    unsigned int* num_found_results_raw = thrust::raw_pointer_cast(num_found_results_dev.data());
    unsigned int* intersection_candidates = thrust::raw_pointer_cast(buffer_results_dev.data());
    unsigned int* first_query_result_raw = thrust::raw_pointer_cast(first_query_result.data());
    
    // get number of  intersection candidates
    compute_num_of_query_result_kernel<<<(num_faces + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(F_d_raw, bvh_dev, num_found_results_raw, num_faces);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, num_found_results_raw, num_found_results_raw + num_faces + 1, first_query_result_raw);
    cudaDeviceSynchronize();

    // save data of intersection candidates
    compute_query_list_kernel<<<(num_faces + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(F_d_raw, bvh_dev, num_found_results_raw, first_query_result_raw, intersection_candidates, num_faces);
    cudaDeviceSynchronize();

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    printf("query runtime %f ms\n", milliseconds1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);


    // actual triangle intersection test based on query result ---------------------------------------
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    // list of actual intersection pairs
    const int maxIntersections = num_faces;
    unsigned int* d_intersections;
    cudaMalloc((void**)&d_intersections, 2 * maxIntersections * sizeof(unsigned int));
    
    // tmp variables for atomic operation
    unsigned int* d_pos;
    cudaMalloc(&d_pos, sizeof(unsigned int));
    cudaMemset(d_pos, 0, sizeof(unsigned int));

    // isIntersect
    unsigned int h_isIntersect = 0;
    unsigned int* d_isIntersect;
    cudaMalloc((void**)&d_isIntersect, sizeof(unsigned int));
    cudaMemcpy(d_isIntersect, &h_isIntersect, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // get query result size
    unsigned int num_query_result = 0;
    cudaMemcpy(&num_query_result, &first_query_result_raw[num_faces], sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // actual number of query result without the query idx itself
    num_query_result /= 2;

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(0),
                     thrust::make_counting_iterator<unsigned int>(num_query_result),
                     [epsilon, d_isIntersect, d_pos, d_intersections, maxIntersections, first_query_result_raw, triangles_d_raw, num_found_results_raw, intersection_candidates, F_d_raw] __device__(std::size_t idx) {
                         
                        unsigned int query_idx = intersection_candidates[2 * idx];
                        unsigned int current_idx = intersection_candidates[2 * idx + 1];
                        const unsigned int num_found = num_found_results_raw[query_idx];
                        
                        // invalid faces
                        if(num_found == 0) return;
                        if(query_idx == 0xFFFFFFFF) return;
                        if(current_idx == 0xFFFFFFFF) return;

                        // Retrieve faces for idx and query_idx
                        Triangle<int> current_face = F_d_raw[current_idx];
                        Triangle<int> query_face = F_d_raw[query_idx];

                        Face<float3> current_tris = triangles_d_raw[current_idx];
                        Face<float3> query_tris = triangles_d_raw[query_idx];

                        int vertices_current[] = {current_face.i, current_face.j, current_face.k};
                        int vertices_query[] = {query_face.i, query_face.j, query_face.k};

                        int num_count = 0;

                        float3 p1,q1,r1,p2,q2,r2;
                        p1 = current_tris.v0;
                        q1 = current_tris.v1;
                        r1 = current_tris.v2;
                        p2 = query_tris.v0;
                        q2 = query_tris.v1;
                        r2 = query_tris.v2;

                        // remove invalid face from test
                        // zero faces
                        if(detect_zero_face(p1, q1, r1)) return;
                        if(detect_zero_face(p2, q2, r2)) return;
                        // line
                        if(detect_invalid_face(p1,q1,r1)) return;
                        if(detect_invalid_face(p2,q2,r2)) return;
                        // same plane
                        if(detect_same_plane_coord(p1,q1,r1,p2,q2,r2)) return;
                        // line
                        if(vertices_query[0]== vertices_query[1] || vertices_query[1] == vertices_query[2] || vertices_query[2] == vertices_query[0]) return;
                        if(vertices_current[0]== vertices_current[1] || vertices_current[1] == vertices_current[2] || vertices_current[2] == vertices_current[0]) return;

                        float distP1 = norm3df(p1.x, p1.y, p1.z);
                        float distQ1 = norm3df(q1.x, q1.y, q1.z);
                        float distR1 = norm3df(r1.x, r1.y, r1.z);
                        float distP2 = norm3df(p2.x, p2.y, p2.z);
                        float distQ2 = norm3df(q2.x, q2.y, q2.z);
                        float distR2 = norm3df(r2.x, r2.y, r2.z);
                        float avgDist = (distP1 + distQ1 + distR1 + distP2 + distQ2 + distR2) / (10.0f * 6.0f);
                        //scaling
                        p1.x = p1.x / avgDist; p1.y = p1.y / avgDist; p1.z = p1.z / avgDist;
                        q1.x = q1.x / avgDist; q1.y = q1.y / avgDist; q1.z = q1.z / avgDist;
                        r1.x = r1.x / avgDist; r1.y = r1.y / avgDist; r1.z = r1.z / avgDist;
                        
                        p2.x /= avgDist; p2.y /= avgDist; p2.z /= avgDist;
                        q2.x /= avgDist; q2.y /= avgDist; q2.z /= avgDist;
                        r2.x /= avgDist; r2.y /= avgDist; r2.z /= avgDist;

                        for(unsigned int j = 0; j < 3; j++){
                            int vertex_current = vertices_current[j];
                            
                            for(unsigned int k = 0; k < 3; k++){
                                if(vertex_current == vertices_query[k]){
                                    num_count++;
                                }
                            }
                        }

                            float tri_a[3][3];
                            float tri_b[3][3];
                            copy_v3_v3_float_float3(tri_a[0], p1);
                            copy_v3_v3_float_float3(tri_a[1], q1);
                            copy_v3_v3_float_float3(tri_a[2], r1);
                            copy_v3_v3_float_float3(tri_b[0], p2);
                            copy_v3_v3_float_float3(tri_b[1], q2);
                            copy_v3_v3_float_float3(tri_b[2], r2);
                            // check if coplanar
                            if(is_coplanar_blender(tri_a, tri_b)){
                                //printf("coplanar\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nf 1 2 3\nf 4 5 6\n", p1.x, p1.y, p1.z, q1.x, q1.y, q1.z, r1.x, r1.y, r1.z, p2.x, p2.y, p2.z, q2.x, q2.y, q2.z, r2.x, r2.y, r2.z);
                                // no sharing
                                if(num_count == 0){
                                    //printf("case 0 idx %d query %d\n",idx,query_idx);
                                    if(coplanar_without_sharing_test(p1,q1,r1,p2,q2,r2)){
                                        atomicExch(d_isIntersect, 1);
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                    }
                                    return;
                                }
                                
                                // vertex sharing
                                if(num_count == 1){
                                    if(coplanar_vertex_sharing_test(p1,q1,r1,p2,q2,r2,epsilon)){
                                        atomicExch(d_isIntersect, 1);
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                    }
                                    return;
                                }
                                // edge sharing
                                if(num_count == 2){
                                    //printf("case 2 idx %d query %d\n",idx,query_idx);
                                    // if other vertex is same side (intersect)
                                    if(coplanar_same_side_test(p1,q1,r1,p2,q2,r2,epsilon)){
                                        atomicExch(d_isIntersect, 1);
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                    }
                                    return;
                                }
                                // identical face (always intersect)
                                if(num_count == 3){
                                    // printf("case 3 idx %d query %d\n",idx,query_idx);
                                    // printf("case 3 passed idx %d query %d\n",idx,query_idx);
                                    atomicExch(d_isIntersect, 1);
                                    int pos = atomicAdd(d_pos, 2);
                                    if (pos < 2 * maxIntersections - 2) {
                                        d_intersections[pos] = query_idx;
                                        d_intersections[pos + 1] = current_idx;
                                    }
                                    return;
                                }
                                return;
                            }
                            else{
                                // no coplanar, shared edge
                                if(num_count == 2){
                                    return; // remove from the test
                                }
                                else if(detect_shared_edge_coord(p1,q1,r1,p2,q2,r2)){ // remove from the test
                                    return;
                                }

                                float3 source, target;

                                source = make_float3(1,1,1);
                                target = make_float3(-1,-1,-1);

                                float r_i1[3];
                                float r_i2[3];
                                // actual intersection test
                                bool isIntersecting = isect_tri_tri_v3(p1,q1,r1,p2,q2,r2,r_i1,r_i2);
                                
                                if(isIntersecting){
                                    copy_v3_v3_float3_float(source, r_i1);
                                    copy_v3_v3_float3_float(target, r_i2);
                                    float dist = largest_distance(source, target);
                                    bool sharedVertex = (num_count == 1);
                                    // if the distance is less than eps with shared vertex, the intersection point would be shared vertex
                                    if(dist < epsilon && sharedVertex){
                                        return; // not self intersect
                                    }
                                    else{
                                        atomicExch(d_isIntersect, 1);
                                        //printf("intersect dist %.30f, epsilon %.30f\nsource v %.30f %.30f %.30f\ntarget v %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nv %.30f %.30f %.30f\nf 1 2 3\nf 4 5 6\n", dist, epsilon, source.x, source.y, source.z, target.x, target.y, target.z, p1.x, p1.y, p1.z, q1.x, q1.y, q1.z, r1.x, r1.y, r1.z, p2.x, p2.y, p2.z, q2.x, q2.y, q2.z, r2.x, r2.y, r2.z);
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                    }
                                }
                                return;             
                                
                            }
                        
                        return;
                     });


    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("tri-tri intersection test %f ms\n", milliseconds2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //print where is intersection ------------
    // Remove if you don't need this
    // unsigned int h_pos;
    // cudaMemcpy(&h_pos, d_pos, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // printf("# of intersection : %d\n", h_pos);
    // unsigned int* h_intersections = new unsigned int[2 * maxIntersections];
    // cudaMemcpy(h_intersections, d_intersections, 2 * maxIntersections * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < h_pos; i += 2) {    
    //     printf("Intersection between triangle %d and triangle %d.\n", 
    //     h_intersections[i], h_intersections[i + 1]);
    // }
    // delete [] h_intersections;
    ////----------------------------



    cudaFree(d_intersections);
    cudaFree(d_pos);

    cudaMemcpy(&h_isIntersect, d_isIntersect, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_isIntersect);
    bvh.clear();

    // free Thrust device_vector
    V_d.clear();
    F_d.clear();
    triangles_d.clear();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total self-intersect test GPU runtime %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return h_isIntersect;
    }
}

#endif