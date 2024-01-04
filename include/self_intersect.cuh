#ifndef LBVH_SELF_INTERSECT_CUH
#define LBVH_SELF_INTERSECT_CUH
#include <vector>
#include <iostream>
#include "bvh.cuh"
#include "query.cuh"
#include "types.cuh"
#include "tri_tri_intersect.cuh"
#define BLOCK_SIZE 64

using namespace std;

namespace selfx{

    __global__ void compute_num_of_query_result_kernel(
        Triangle<int>* F_d_raw,
        lbvh::bvh_device<float, Face<float3>> bvh_dev,
        unsigned int* num_found_query_raw,
        std::size_t num_faces)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_faces) return;
        //unsigned int num_found = 0;
        //if (idx != 23) return;
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
        // printf("minX : %f\n", minX);
        // printf("minY : %f\n", minY);
        // printf("minZ : %f\n", minZ);
        // printf("maxX : %f\n", maxX);
        // printf("maxY : %f\n", maxY);
        // printf("maxZ : %f\n", maxZ);

        //printf("idx %d\n", idx);
        // Perform the query
        unsigned int num_found = lbvh::get_number_of_intersect_candidates(bvh_dev, lbvh::overlaps(query_box), buffer, idx);

        // Copy results to the device vector
        num_found_query_raw[idx] = 2 * num_found;
        //printf("%d\n", num_found_query_raw[idx]);
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
        //return;
        if (idx >= num_faces) return;
        //unsigned int num_found = 0;
        //if (idx != 23) return;
        // Check if the index is valid
        if (F_d_raw[idx].i == -1) {
            //num_found_query_raw[idx] = 0;// itself
            return;
        }

        // 여기서 result 전체를 query에 넣고, first, last를 arg로 주자
        //int buffer_size = num_found_query_raw[idx];
        //unsigned int buffer[BUFFER_SIZE];

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
        // printf("minX : %f\n", minX);
        // printf("minY : %f\n", minY);
        // printf("minZ : %f\n", minZ);
        // printf("maxX : %f\n", maxX);
        // printf("maxY : %f\n", maxY);
        // printf("maxZ : %f\n", maxZ);

        //printf("idx %d\n", idx);
        // Compute the query
        int first = first_query_result_raw[idx];
        if(idx== num_faces-1) printf("first_query %d\n", first_query_result_raw[idx+1]);
        unsigned int num_found = lbvh::query_device(bvh_dev, lbvh::overlaps(query_box), buffer_results_query_raw, idx, first);
    }

    bool self_intersect(Vertex<float> *V, Triangle<int> *F, unsigned int num_vertices, unsigned int num_faces, float epsilon) {
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // init triangle list ----------------------
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
    //const std::size_t N = triangles.size();
    thrust::device_vector<unsigned int> num_found_results_dev(num_faces);
    thrust::device_vector<unsigned int> buffer_results_dev(num_faces * BUFFER_SIZE, 0xFFFFFFFF);
    thrust::device_vector<unsigned int> first_query_result(num_faces + 1);

    unsigned int* num_found_results_raw = thrust::raw_pointer_cast(num_found_results_dev.data());
    unsigned int* buffer_results_raw = thrust::raw_pointer_cast(buffer_results_dev.data());
    unsigned int* first_query_result_raw = thrust::raw_pointer_cast(first_query_result.data());
    
    compute_num_of_query_result_kernel<<<(num_faces + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(F_d_raw, bvh_dev, num_found_results_raw, num_faces);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, num_found_results_raw, num_found_results_raw + num_faces + 1, first_query_result_raw);
    cudaDeviceSynchronize();

    compute_query_list_kernel<<<(num_faces + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(F_d_raw, bvh_dev, num_found_results_raw, first_query_result_raw, buffer_results_raw, num_faces);
    cudaDeviceSynchronize();

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    printf("query runtime %f ms\n", milliseconds1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);


    // remove adjacent faces ---------------------------------------
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    const int maxIntersections = num_faces;
    unsigned int* d_intersections;
    cudaMalloc((void**)&d_intersections, 2 * maxIntersections * sizeof(unsigned int));
    
    unsigned int* d_pos;
    cudaMalloc(&d_pos, sizeof(unsigned int));
    cudaMemset(d_pos, 0, sizeof(unsigned int));

    unsigned int h_isIntersect = 0; // host에서 사용할 변수
    unsigned int* d_isIntersect;        // device에서 사용할 포인터
    cudaMalloc((void**)&d_isIntersect, sizeof(unsigned int));
    cudaMemcpy(d_isIntersect, &h_isIntersect, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    unsigned int num_query_result = 0;
    cudaMemcpy(&num_query_result, &first_query_result_raw[num_faces], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    num_query_result /= 2;
    //printf("num_query_result %d\n", num_query_result);

    // unsigned int* buffer_result_host = new unsigned int[2 * num_query_result];
    // cudaMemcpy(buffer_result_host, buffer_results_raw, 2 * num_query_result * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // for(unsigned int i = 0; i < num_query_result; i++){
    //     printf("i : %d result %d %d\n", i, buffer_result_host[2 * i], buffer_result_host[2 * i + 1]);
    // }


    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(0),
                     thrust::make_counting_iterator<unsigned int>(num_query_result),
                     [epsilon, d_isIntersect, d_pos, d_intersections, maxIntersections, first_query_result_raw, triangles_d_raw, num_found_results_raw, buffer_results_raw, F_d_raw] __device__(std::size_t idx) {
                         
                        unsigned int query_idx = buffer_results_raw[2 * idx];
                        unsigned int current_idx = buffer_results_raw[2 * idx + 1];
                        const unsigned int num_found = num_found_results_raw[query_idx];
                        if(num_found == 0) return;

                            //unsigned int query_idx = buffer_results_raw[i];
                            //if(idx == 0) printf("query_idx %d\n",query_idx);
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
                                        //여기 주석
                                    }
                                }
                            }
                            // if(num_count == 1 && is_coplanar(p1,q1,r1,p2,q2,r2,epsilon)){
                            //     buffer_results_raw[idx * BUFFER_SIZE + i] = 0xFFFFFFFF;
                            // }

                            if(is_coplanar(p1, q1, r1, p2, q2, r2, epsilon)){
                                // no sharing
                                if(num_count == 0){
                                    //printf("case 0 idx %d query %d\n",idx,query_idx);
                                    if(coplanar_without_sharing_test(p1,q1,r1,p2,q2,r2)){
                                        //printf("case 0 passed idx %d query %d\n",idx,query_idx);
                                        // printf("current face1 vertex x %f\n", p1.x);
                                        // printf("current face1 vertex y %f\n", p1.y);
                                        // printf("current face1 vertex z %f\n", p1.z);
                                        // printf("current face2 vertex x %f\n", q1.x);
                                        // printf("current face2 vertex y %f\n", q1.y);
                                        // printf("current face2 vertex z %f\n", q1.z);
                                        // printf("current face3 vertex x %f\n", r1.x);
                                        // printf("current face3 vertex y %f\n", r1.y);
                                        // printf("current face3 vertex z %f\n", r1.z);
                                        // printf("query face1 vertex x %f\n", p2.x);
                                        // printf("query face1 vertex y %f\n", p2.y);
                                        // printf("query face1 vertex z %f\n", p2.z);
                                        // printf("query face2 vertex x %f\n", q2.x);
                                        // printf("query face2 vertex y %f\n", q2.y);
                                        // printf("query face2 vertex z %f\n", q2.z);
                                        // printf("query face3 vertex x %f\n", r2.x);
                                        // printf("query face3 vertex y %f\n", r2.y);
                                        // printf("query face3 vertex z %f\n", r2.z);
                                        atomicExch(d_isIntersect, 1);
                                        // check where is intersection -------------------
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                    }
                                    //buffer_results_raw[idx * BUFFER_SIZE + i] = 0xFFFFFFFF;
                                }
                                
                                // vertex sharing
                                if(num_count == 1){
                                    //printf("case 1 idx %d query %d\n",idx,query_idx);
                                    // check done
                                    if(coplanar_vertex_sharing_test(p1,q1,r1,p2,q2,r2,epsilon)){
                                        // printf("case1 idx %d\n", idx);
                                        // printf("case 1 passed idx %d query %d\n",idx,query_idx);
                                        // printf("current face1 vertex %f, %f, %f;\n", p1.x, p1.y, p1.z);
                                        // printf("current face2 vertex %f, %f, %f;\n", q1.x, q1.y, q1.z);
                                        // printf("current face3 vertex %f, %f, %f;\n", r1.x, r1.y, r1.z);
                                        // printf("query face1 vertex %f, %f, %f;\n", p2.x, p2.y, p2.z);
                                        // printf("query face2 vertex %f, %f, %f;\n", q2.x, q2.y, q2.z);
                                        // printf("query face3 vertex %f, %f, %f;\n", r2.x, r2.y, r2.z);
                                        atomicExch(d_isIntersect, 1);
                                        // check where is intersection -------------------
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                    }
                                    //buffer_results_raw[idx * BUFFER_SIZE + i] = 0xFFFFFFFF;
                                }
                                // edge sharing
                                if(num_count == 2){
                                    //printf("case 2 idx %d query %d\n",idx,query_idx);
                                    // if other vertex is same side (intersect)
                                    if(coplanar_same_side_test(p1,q1,r1,p2,q2,r2,epsilon)){
                                        // printf("case 2 passed idx %d query %d\n",idx,query_idx);
                                        // printf("case2 idx %d\n", idx);
                                        // printf("current face1 vertex %f, %f, %f;\n", p1.x, p1.y, p1.z);
                                        // printf("current face2 vertex %f, %f, %f;\n", q1.x, q1.y, q1.z);
                                        // printf("current face3 vertex %f, %f, %f;\n", r1.x, r1.y, r1.z);
                                        // printf("query face1 vertex %f, %f, %f;\n", p2.x, p2.y, p2.z);
                                        // printf("query face2 vertex %f, %f, %f;\n", q2.x, q2.y, q2.z);
                                        // printf("query face3 vertex %f, %f, %f;\n", r2.x, r2.y, r2.z);
                                        atomicExch(d_isIntersect, 1);
                                        // check where is intersection -------------------
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                    }
                                    // remove faces from query test (already tested)
                                    //buffer_results_raw[idx * BUFFER_SIZE + i] = 0xFFFFFFFF;
                                }
                                // identical face (always intersect)
                                if(num_count == 3){
                                    // printf("case 3 idx %d query %d\n",idx,query_idx);
                                    // printf("case 3 passed idx %d query %d\n",idx,query_idx);
                                    atomicExch(d_isIntersect, 1);
                                    // check where is intersection -------------------
                                    int pos = atomicAdd(d_pos, 2);
                                    if (pos < 2 * maxIntersections - 2) {
                                        d_intersections[pos] = query_idx;
                                        d_intersections[pos + 1] = current_idx;
                                    }
                                    //buffer_results_raw[idx * BUFFER_SIZE + i] = 0xFFFFFFFF;
                                }
                                buffer_results_raw[2 * idx + 1] = 0xFFFFFFFF;
                            }
                            else{
                                // no coplanar, share edge
                                if(num_count == 2){
                                    //buffer_results_raw[idx * BUFFER_SIZE + i] = 0xFFFFFFFF;
                                    buffer_results_raw[2 * idx + 1] = 0xFFFFFFFF;
                                }
                            }
                        
                        return;
                     });


    // cudaMemcpy(buffer_result_host, buffer_results_raw, 2 * num_query_result * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // for(unsigned int i = 0; i < num_query_result; i++){
    //     printf("i : %d result %d %d\n", i, buffer_result_host[2 * i], buffer_result_host[2 * i + 1]);
    // }                     

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("remove adjacent faces %f ms\n", milliseconds2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);


    // actual triangle - triangle intersection test with query list --------------------------
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);

    
    thrust::for_each(thrust::device,
                    thrust::make_counting_iterator<std::size_t>(0),//0
                    thrust::make_counting_iterator<std::size_t>(num_query_result),//N
                    //[d_sources_targets, epsilon, d_pos, F_d_raw, buffer_results_raw, num_found_results_raw, triangles_d_raw, d_isIntersect, d_intersections, maxIntersections, d_invalid_indices, d_invalid_count] __device__(std::size_t idx) {
                    [epsilon, d_pos, F_d_raw, first_query_result_raw, buffer_results_raw, num_found_results_raw, triangles_d_raw, d_isIntersect, d_intersections, maxIntersections] __device__(std::size_t idx) {
                        // invalid triangle
                        //if(idx != 7) return;
                        unsigned int query_idx = buffer_results_raw[2 * idx];
                        unsigned int current_idx = buffer_results_raw[2 * idx + 1];
                        
                        if(query_idx == 0xFFFFFFFF) return;
                        if(current_idx == 0xFFFFFFFF) return;

                        //printf("query idx %u current idx %u\n", query_idx, current_idx);
                        if(F_d_raw[query_idx].i == -1) {
                          //unsigned int pos = atomicAdd(d_invalid_count, 1);
                          //d_invalid_indices[pos] = idx;
                          return;
                        }
                        if(F_d_raw[current_idx].i == -1) return; // invalid triangle

                        bool coplanar;
                        //unsigned int num_found = num_found_results_raw[query_idx];
                        float3 source, target;

                        source = make_float3(1,1,1);
                        target = make_float3(-1,-1,-1);

                        float3 p1 = triangles_d_raw[query_idx].v0;
                        float3 q1 = triangles_d_raw[query_idx].v1;
                        float3 r1 = triangles_d_raw[query_idx].v2;

                        float distP1 = norm3df(p1.x, p1.y, p1.z);
                        float distQ1 = norm3df(q1.x, q1.y, q1.z);
                        float distR1 = norm3df(r1.x, r1.y, r1.z);

                            
                            float3 p2 = triangles_d_raw[current_idx].v0;
                            float3 q2 = triangles_d_raw[current_idx].v1;
                            float3 r2 = triangles_d_raw[current_idx].v2;

                            float distP2 = norm3df(p2.x, p2.y, p2.z);
                            float distQ2 = norm3df(q2.x, q2.y, q2.z);
                            float distR2 = norm3df(r2.x, r2.y, r2.z);


                            // compute average distance
                            float avgDist = (distP1 + distQ1 + distR1 + distP2 + distQ2 + distR2) / (10.0f * 6.0f);

                            // scale
                            p1.x /= avgDist; p1.y /= avgDist; p1.z /= avgDist;
                            q1.x /= avgDist; q1.y /= avgDist; q1.z /= avgDist;
                            r1.x /= avgDist; r1.y /= avgDist; r1.z /= avgDist;

                            p2.x /= avgDist; p2.y /= avgDist; p2.z /= avgDist;
                            q2.x /= avgDist; q2.y /= avgDist; q2.z /= avgDist;
                            r2.x /= avgDist; r2.y /= avgDist; r2.z /= avgDist;

                            // printf("p1: %f, %f, %f\n", p1.x, p1.y, p1.z);
                            // printf("q1: %f, %f, %f\n", q1.x, q1.y, q1.z);
                            // printf("r1: %f, %f, %f\n", r1.x, r1.y, r1.z);

                            // printf("p2: %f, %f, %f\n", p2.x, p2.y, p2.z);
                            // printf("q2: %f, %f, %f\n", q2.x, q2.y, q2.z);
                            // printf("r2: %f, %f, %f\n", r2.x, r2.y, r2.z);


                            bool isIntersecting = tri_tri_intersection_test_3d(p1, q1, r1, p2, q2, r2, coplanar, source, target);
                            // printf("idx %d source x %f y %f z %f\n",idx, source.x, source.y, source.z);
                            // printf("idx %d target x %f y %f z %f\n",idx, target.x, target.y, target.z);
                            float dist = largest_distance(source, target);
                            bool sharedVertex = check_if_shared_vertex(p1,q1,r1,p2,q2,r2);
                            if(dist < epsilon && sharedVertex){
                                return; // not self intersect
                            }
                            else{
                                if(isIntersecting) {
                                    atomicExch(d_isIntersect, 1);
                                    // check where is intersection -------------------
                                    //printf("source x %f, y %f, z %f target x %f, y %f, z %f\n",source.x, source.y, source.z, target.x, target.y, target.z);
                                    int pos = atomicAdd(d_pos, 2);
                                    if (pos < 2 * maxIntersections - 2) {
                                        d_intersections[pos] = query_idx;
                                        d_intersections[pos + 1] = current_idx;
                                    }
                                }
                            }
                        return;
                    });
    cudaDeviceSynchronize();

    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start3, stop3);
    printf("actual tri-tri test %f ms\n", milliseconds3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //print where is intersection ------------
    // Remove if you don't need this
    unsigned int h_pos;
    cudaMemcpy(&h_pos, d_pos, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("# of intersection : %d\n", h_pos);
    unsigned int* h_intersections = new unsigned int[2 * maxIntersections];
    cudaMemcpy(h_intersections, d_intersections, 2 * maxIntersections * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_pos; i += 2) {    
        printf("Intersection between triangle %d and triangle %d.\n", 
        h_intersections[i], h_intersections[i + 1]);
    }
    delete [] h_intersections;
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