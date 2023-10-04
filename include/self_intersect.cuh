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
    bool self_intersect(const vector<float> &V, const vector<unsigned int> &F) {
    
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
    cudaEvent_t start_init, stop_init;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventRecord(start_init);

    unsigned int num_vertices = V.size()/3;
    unsigned int num_faces = F.size()/3;

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

    lbvh::bvh<float, Triangle, aabb_getter> bvh(triangles_d.begin(), triangles_d.end(), false);
    // get device ptr
    const auto bvh_dev = bvh.get_device_repr();
    
    cudaEventRecord(stop0);
    cudaEventSynchronize(stop0);
    float milliseconds0 = 0;
    cudaEventElapsedTime(&milliseconds0, start0, stop0);
    printf("bvh constructor call runtime %f ms\n", milliseconds0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    


    std::cout << "testing query_device:overlap ...\n";


    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    //const std::size_t N = triangles.size();
    thrust::device_vector<unsigned int> num_found_results_dev(num_faces);
    thrust::device_vector<unsigned int> buffer_results_dev(num_faces * BUFFER_SIZE, 0xFFFFFFFF);

    

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
                     thrust::make_counting_iterator<std::size_t>(num_faces),
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
    
    //copy result to host vector
    /*
    std::vector<unsigned int> num_found_results_h(num_faces);
    thrust::copy(num_found_results_dev.begin(), num_found_results_dev.end(), num_found_results_h.begin());
    std::vector<unsigned int> buffer_results_h(num_faces * BUFFER_SIZE);
    thrust::copy(buffer_results_dev.begin(), buffer_results_dev.end(), buffer_results_h.begin());
    cudaDeviceSynchronize();*/
    
    /*
    for(int i = 0; i<num_faces; i++){
        std::cout<<"i : "<<i<<" num_found : "<<num_found_results_h[i]<<std::endl;
    }


    for(int i = 0; i<num_faces; i++){
                    cout<<"i : "<< i<<endl;
        for(int j = 0; j<BUFFER_SIZE; j++){
            cout<<buffer_results_h[i*BUFFER_SIZE + j]<<" ";
        }
        cout<<endl;
    }*/

    // delete adjacent faces
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(0),
                     thrust::make_counting_iterator<unsigned int>(num_faces),
                     [num_found_results_raw, buffer_results_raw, F_d_raw] __device__(std::size_t idx) {
                         
                        const unsigned int num_found = num_found_results_raw[idx];

                        for(unsigned int i = 0; i < num_found; i++){
                            unsigned int query_idx = buffer_results_raw[idx * BUFFER_SIZE + i];
                            bool flag = false;
                            for(unsigned int j = 0; j < 3; j++){
                                unsigned int face_current = F_d_raw[idx * 3 + j];
                                for(unsigned int k = 0; k < 3; k++){
                                    if(face_current == F_d_raw[query_idx * 3 + k]){
                                        flag = true;
                                        buffer_results_raw[idx * BUFFER_SIZE + i] = 0xFFFFFFFF;
                                        break;
                                    }
                                }
                                if(flag) break;
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


    // Copy results to host and print
    /*
    std::vector<unsigned int> num_found_results_h1(num_faces);
    thrust::copy(num_found_results_dev.begin(), num_found_results_dev.end(), num_found_results_h1.begin());
    std::vector<unsigned int> buffer_results_h1(num_faces * BUFFER_SIZE);
    thrust::copy(buffer_results_dev.begin(), buffer_results_dev.end(), buffer_results_h1.begin());

    for (unsigned int i = 0; i < num_faces; i++) {
        std::cout << "i : " << i << " num_found : " << num_found_results_h1[i] << std::endl;
    }
    for (unsigned int i = 0; i < num_faces; i++) {
        std::cout << "i : " << i << " buffer: ";
        for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
            std::cout << buffer_results_h1[i * BUFFER_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }*/

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);

    
    bool isIntersect = lbvh::tri_tri_intersect(triangles_d_raw, buffer_results_raw, num_found_results_raw, num_faces, BUFFER_SIZE);
    
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start3, stop3);
    printf("actual tri-tri test %f ms\n", milliseconds3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    //end_cpu = clock();

    //cout << "CPU runtime : " << (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC << " sec\n";

    //cout<<"tri_tri intersect : "<<isIntersect<<endl;
    bvh.clear();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total self-intersect test GPU runtime %f ms\n", milliseconds);
    
    return isIntersect;
    }

}

#endif