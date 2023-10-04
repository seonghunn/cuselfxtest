#ifndef LBVH_SELF_INTERSECT_CUH
#define LBVH_SELF_INTERSECT_CUH
#include <vector>
#include <iostream>
#include "bvh.cuh"
#include "query.cuh"
#include "types.h"
#include "tri_tri_intersect.cuh"
//#define BUFFER_SIZE 64

using namespace std;

namespace lbvh{
    bool self_intersect(const vector<float> &V, const vector<unsigned int> &F, thrust::device_vector<unsigned int>& adj_faces_dev) {
    
    /*
    // 1. Create a host vector of appropriate size.
    std::vector<unsigned int> host_output(num_faces * FACES_SIZE);

    // 2. Copy data from device to host.
    thrust::copy(adj_faces_dev.begin(), adj_faces_dev.end(), host_output.begin());

    // 3. Print the contents.
    for (std::size_t i = 0; i < num_faces; i++) {
        std::cout << "cface : " << i << " adjacent faces: ";
        for (std::size_t j = 0; j < FACES_SIZE; j++) {
            unsigned int face_idx = host_output[i * FACES_SIZE + j];
            if (face_idx != 0xFFFFFFFF) { // Check for non-NULL values
                std::cout << face_idx << " ";
            }
        }
        std::cout << std::endl;
    }
*/
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // init triangle list ----------------------
    unsigned int num_vertices = V.size()/3;
    unsigned int num_faces = F.size()/3;

    std::vector<Triangle> triangles;
    thrust::device_vector<float> V_d = V;
    thrust::device_vector<unsigned int> F_d = F;
    thrust::device_vector<Triangle> triangles_d(num_faces);

    float* V_d_raw = thrust::raw_pointer_cast(V_d.data());
    unsigned int* F_d_raw = thrust::raw_pointer_cast(F_d.data());
    Triangle* triangles_d_raw = thrust::raw_pointer_cast(triangles_d.data());


    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator<std::size_t>(num_faces),
                     [V_d_raw, F_d_raw, triangles_d_raw] __device__(std::size_t idx){
                        Triangle tri;
                        unsigned int v0_row = F_d_raw[idx * 3 + 0];
                        unsigned int v1_row = F_d_raw[idx * 3 + 1];
                        unsigned int v2_row = F_d_raw[idx * 3 + 2];
                        tri.v0 = make_float3(V_d_raw[v0_row * 3 + 0], V_d_raw[v0_row * 3 + 1], V_d_raw[v0_row * 3 + 2]);
                        tri.v1 = make_float3(V_d_raw[v1_row * 3 + 0], V_d_raw[v1_row * 3 + 1], V_d_raw[v1_row * 3 + 2]);
                        tri.v2 = make_float3(V_d_raw[v2_row * 3 + 0], V_d_raw[v2_row * 3 + 1], V_d_raw[v2_row * 3 + 2]);
                        triangles_d_raw[idx] = tri;

                        return;
                     });


    // 1. 디바이스에서 호스트로 복사
std::vector<Triangle> triangles_h(num_faces);
thrust::copy(triangles_d.begin(), triangles_d.end(), triangles_h.begin());

// 2. 3개의 삼각형 단위로 출력
for (size_t i = 0; i < triangles_h.size(); i += 3) {
    for (size_t j = 0; j < 3 && (i + j) < triangles_h.size(); ++j) {
        const Triangle& tri = triangles_h[i + j];
        std::cout << "Triangle " << (i + j + 1) << ": "
                  << "v0(" << tri.v0.x << ", " << tri.v0.y << ", " << tri.v0.z << ") "
                  << "v1(" << tri.v1.x << ", " << tri.v1.y << ", " << tri.v1.z << ") "
                  << "v2(" << tri.v2.x << ", " << tri.v2.y << ", " << tri.v2.z << ")"
                  << std::endl;
    }
    std::cout << std::endl;
}

        // V와 F로부터 삼각형(Triangle) 목록을 초기화합니다.

    for (unsigned int i = 0; i < F.size()/3; i++)
    {
        Triangle tri;
        unsigned int v0_row = F[i * 3 + 0];
        unsigned int v1_row = F[i * 3 + 1];
        unsigned int v2_row = F[i * 3 + 2];
        tri.v0 = make_float3(V[v0_row * 3 + 0], V[v0_row * 3 + 1], V[v0_row * 3 + 2]);
        tri.v1 = make_float3(V[v1_row * 3 + 0], V[v1_row * 3 + 1], V[v1_row * 3 + 2]);
        tri.v2 = make_float3(V[v2_row * 3 + 0], V[v2_row * 3 + 1], V[v2_row * 3 + 2]);
        triangles.push_back(tri);
    }

    for (unsigned int i = 0; i < triangles.size(); i++)
{
    printf("Triangle %d: v0(%f, %f, %f) v1(%f, %f, %f) v2(%f, %f, %f)\n", 
        i + 1, 
        triangles[i].v0.x, triangles[i].v0.y, triangles[i].v0.z,
        triangles[i].v1.x, triangles[i].v1.y, triangles[i].v1.z,
        triangles[i].v2.x, triangles[i].v2.y, triangles[i].v2.z);

    // If i is 2 (0-indexed), 5, 8, ... then insert a newline for better readability
    if ((i + 1) % 3 == 0)
    {
        printf("\n");
    }
}

    cudaEvent_t start0, stop0;
    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);
    cudaEventRecord(start0);

    //construct bvh
    lbvh::bvh<float, Triangle, aabb_getter> bvh(triangles.begin(), triangles.end(), false);
    
    cudaEventRecord(stop0);
    cudaEventSynchronize(stop0);
    float milliseconds0 = 0;
    cudaEventElapsedTime(&milliseconds0, start0, stop0);
    printf("bvh constructor call runtime %f ms\n", milliseconds0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    
    // get device ptr
    const auto bvh_dev = bvh.get_device_repr();

    std::cout << "testing query_device:overlap ...\n";


    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    const std::size_t N = triangles.size();
    thrust::device_vector<unsigned int> num_found_results_dev(N);
    thrust::device_vector<unsigned int> buffer_results_dev(N * BUFFER_SIZE, 0xFFFFFFFF);

    

    //Thrust의 device_vector는 host 코드에서 사용하기 위한 래퍼(wrapper)입니다.
    //즉, device_vector 자체는 GPU 메모리에 대한 직접적인 접근을 제공하는 것이 아니라, GPU 메모리 관리와 데이터 전송에 대한 API를 제공합니다.
    //실제 GPU 연산은 raw pointer를 사용해야 합니다.
    //thrust device vector는 host에서 메소드를 잘 사용할 수 있도록 해주고, 그 메소드의 수행은 내부적으로 gpu에서 동작
    //따라서 메소드가 아닌 벡터 자체에 접근하려면 raw pointer를 추출 (CUDA 커널에서 사용하는 등의)
    // device vector to store query result
    // query result : 0xFFFFFFFF에 도달하면 끝

    // make query list
    unsigned int* num_found_results_raw = thrust::raw_pointer_cast(num_found_results_dev.data());
    unsigned int* buffer_results_raw = thrust::raw_pointer_cast(buffer_results_dev.data());

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator<std::size_t>(N),
                     [bvh_dev, num_found_results_raw, buffer_results_raw] __device__(std::size_t idx)
                     {
                        unsigned int buffer[BUFFER_SIZE];
                        // get query object
                        const auto self = bvh_dev.objects[idx];

                        // set query AABB
                        lbvh::aabb<float> query_box;
                        float minX = min(self.v0.x, min(self.v1.x, self.v2.x));
                        float minY = min(self.v0.y, min(self.v1.y, self.v2.y));
                        float minZ = min(self.v0.z, min(self.v1.z, self.v2.z));

                        float maxX = max(self.v0.x, max(self.v1.x, self.v2.x));
                        float maxY = max(self.v0.y, max(self.v1.y, self.v2.y));
                        float maxZ = max(self.v0.z, max(self.v1.z, self.v2.z));

                        query_box.lower = make_float4(minX, minY, minZ, 0);
                        query_box.upper = make_float4(maxX, maxY, maxZ, 0);
                        
                        // num found : # of aabb that intersects
                        // buffer : list of object index that intersects
                        const unsigned int num_found = lbvh::query_device(
                            bvh_dev, lbvh::overlaps(query_box), buffer, idx);

                        // copy to device vector
                        num_found_results_raw[idx] = num_found;
                        for (int i = 0; i < num_found; i++)
                        {
                            buffer_results_raw[idx * BUFFER_SIZE + i] = buffer[i];
                        }

                        return;
                    });

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    printf("find query runtime %f ms\n", milliseconds1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    //print
    /*
    //copy result to host vector
    std::vector<unsigned int> num_found_results_h(N);
    thrust::copy(num_found_results_dev.begin(), num_found_results_dev.end(), num_found_results_h.begin());
    std::vector<unsigned int> buffer_results_h(N * BUFFER_SIZE);
    thrust::copy(buffer_results_dev.begin(), buffer_results_dev.end(), buffer_results_h.begin());
    cudaDeviceSynchronize();
    
    for(int i = 0; i<N; i++){
        std::cout<<"i : "<<i<<" num_found : "<<num_found_results_h[i]<<std::endl;
    }


    for(int i = 0; i<N; i++){
                    cout<<"i : "<< i<<endl;
        for(int j = 0; j<BUFFER_SIZE; j++){
            cout<<buffer_results_h[i*BUFFER_SIZE + j]<<" ";
        }
        cout<<endl;
    }
*/
    // delete adjacent faces
    // 5. Remove adjacent faces using thrust::for_each.
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    unsigned int* adj_faces_raw = thrust::raw_pointer_cast(adj_faces_dev.data());
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(0),
                     thrust::make_counting_iterator<unsigned int>(N),
                     [adj_faces_raw, num_found_results_raw, buffer_results_raw] __device__(std::size_t idx) {
                         
                        const unsigned int num_found = num_found_results_raw[idx];
                        
                        for (unsigned int i = 0; i < FACES_SIZE; i++) {
                            unsigned int adj_face = adj_faces_raw[idx * FACES_SIZE + i];
                            for (unsigned int j = 0; j < num_found; j++) {
                                //if (adj_faces_raw[idx * FACES_SIZE + i] == buffer_results_raw[idx * BUFFER_SIZE + j]) {
                                if (adj_face == buffer_results_raw[idx * BUFFER_SIZE + j]) {    
                                    buffer_results_raw[idx * BUFFER_SIZE + j] = 0xFFFFFFFF;
                                }
                            }
                        }
                         

                        return;
                     });
                     
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("remove adjacent faces %f ms\n", milliseconds2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

/*
    // 6. Copy results to host and print
    std::vector<unsigned int> num_found_results_h1(N);
    thrust::copy(num_found_results_dev.begin(), num_found_results_dev.end(), num_found_results_h1.begin());
    std::vector<unsigned int> buffer_results_h1(N * BUFFER_SIZE);
    thrust::copy(buffer_results_dev.begin(), buffer_results_dev.end(), buffer_results_h1.begin());

    for (unsigned int i = 0; i < triangles.size(); i++) {
        std::cout << "i : " << i << " num_found : " << num_found_results_h1[i] << std::endl;
    }
    for (unsigned int i = 0; i < triangles.size(); i++) {
        std::cout << "i : " << i << " buffer: ";
        for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
            std::cout << buffer_results_h1[i * BUFFER_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
*/
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);

    
    bool isIntersect = lbvh::tri_tri_intersect(triangles_d_raw, buffer_results_raw, num_found_results_raw, N, BUFFER_SIZE);
    
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start3, stop3);
    printf("actual tri-tri test %f ms\n", milliseconds3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);


    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total self-intersect test GPU runtime %f ms\n", milliseconds);
    //end_cpu = clock();

    //cout << "CPU runtime : " << (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC << " sec\n";

    //cout<<"tri_tri intersect : "<<isIntersect<<endl;
    bvh.clear();
    return isIntersect;
    }

}

#endif