#ifndef LBVH_ADJ_FACES_CUH
#define LBVH_ADJ_FACES_CUH

#include <thrust/for_each.h>
#include <vector>
#include "types.h"
#include <thrust/device_vector.h>

//TODO: init adj faces
using namespace std;
namespace selfx{

    /*
__global__ void computeAdjFaces(const unsigned int* d_F, unsigned int* d_adj_faces, unsigned int numTriangles, unsigned int FACES_SIZE) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;

    for (unsigned int j = 0; j < 3; j++) {
        unsigned int vertexIdx = d_F[idx * 3 + j];
        for (unsigned int k = 0; k < FACES_SIZE; k++) {
            unsigned int slot = vertexIdx * FACES_SIZE + k;
            if (atomicCAS(&d_adj_faces[slot], 0xFFFFFFFF, idx) == 0xFFFFFFFF) {
                // Slot updated successfully, break out
                break;
            }
        }
    }
}

std::vector<unsigned int> parallel_adj_faces(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    unsigned int numVertices = F.cols(); // Assuming F is the face-vertex adjacency matrix
    unsigned int numTriangles = F.rows();
    const unsigned int FACES_SIZE = FACES_SIZE; // You've previously defined FACES_SIZE
    unsigned int num_vertices = V.rows();

    // Allocating and initializing memory on the device
    thrust::device_vector<unsigned int> d_F(F.data(), F.data() + F.FACES_SIZE());
    thrust::device_vector<unsigned int> d_adj_faces(numVertices * FACES_SIZE, 0xFFFFFFFF);

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocks = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    computeAdjFaces<<<blocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_F.data()), 
                                                thrust::raw_pointer_cast(d_adj_faces.data()), 
                                                numTriangles, 
                                                FACES_SIZE);

    // Copy back results to host
    std::vector<unsigned int> adj_faces_host(numVertices * FACES_SIZE);
    thrust::copy(d_adj_faces.begin(), d_adj_faces.end(), adj_faces_host.begin());

    for (unsigned int i = 0; i < num_vertices; ++i) {
        std::cout << "Parallel Vertex " << i << " Parallel adjacent faces: ";
        for (unsigned int j = 0; j < FACES_SIZE; ++j) {
            if (adj_faces_host[i * FACES_SIZE + j] != 0xFFFFFFFF) {
                std::cout << adj_faces_host[i * FACES_SIZE + j] << " ";
            }
        }
        std::cout << std::endl;
    return adj_faces_host;
}
}
*/

void adj_faces(std::vector<float> &V, std::vector<unsigned int> &F, thrust::device_vector<unsigned int>& adj_faces_dev, unsigned int num_faces){
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    thrust::device_vector<float> V_dev(V.begin(), V.end());
    thrust::device_vector<unsigned int> F_dev(F.begin(), F.end());
    
    unsigned int* adj_faces_raw = thrust::raw_pointer_cast(adj_faces_dev.data());
    float* V_dev_raw = thrust::raw_pointer_cast(V_dev.data());
    unsigned int* F_dev_raw = thrust::raw_pointer_cast(F_dev.data());
    unsigned int faces_size = FACES_SIZE;

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(0),
                     thrust::make_counting_iterator<unsigned int>(num_faces),
                     [adj_faces_raw, F_dev_raw, num_faces, faces_size] __device__(std::size_t idx){
                        unsigned int num_found = 0;
                        for(unsigned int a = 0; a <3; a++){
                            for(unsigned int j = 0; j < num_faces; j++){
                                for(unsigned int b = 0; b<3; b++){
                                    if(idx != j && F_dev_raw[idx * 3 + a] == F_dev_raw[j * 3 + b]){
                                        adj_faces_raw[idx * faces_size + num_found++] = j;
                                        break;
                                    }
                                }
                            }
                        }
                        return;
                     });

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the elapsed time
    std::cout << "adj_faces function took: " << elapsedTime << " ms" << std::endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

/*
void adj_faces(Eigen::MatrixXd &V, Eigen::MatrixXi &F, thrust::device_vector<unsigned int>& adj_faces_dev){
    // vertex index to face index mapping
    unsigned int num_vertices = V.rows();
    unsigned int num_faces = F.rows();
    unsigned int* adj_faces_raw = thrust::raw_pointer_cast(adj_faces_dev.data());
    
    cout << "V" << endl;
    cout << V << endl;
    cout << "F" << endl;
    cout << F << endl;

    std::vector<unsigned int> adj_faces_host(num_faces * FACES_SIZE, 0xFFFFFFFF);

    for(unsigned int i = 0; i < F.rows(); i++){
        unsigned int num_found = 0;
        for(unsigned int a = 0; a < 3; a++){
            for(unsigned int j = 0; j < F.rows(); j++){
                for(unsigned int b = 0; b < 3; b++){
                    if(i != j && F(i, a) == F(j, b)){
                        adj_faces_host[i * FACES_SIZE + num_found] = j;
                        num_found++;
                        break;
                    }
                }
            }
        }
    }

    // Copy from the host vector to the device vector.
    thrust::copy(adj_faces_host.begin(), adj_faces_host.end(), adj_faces_dev.begin());

    // Just to print results (Optional)
    for (unsigned int i = 0; i < num_faces; i++) {
        std::cout << "face : " << i << " adjacent faces: ";
        for (unsigned int j = 0; j < FACES_SIZE; j++) {
            unsigned int face_idx = adj_faces_host[i * FACES_SIZE + j];
            if (face_idx != 0xFFFFFFFF) { 
                std::cout << face_idx << " ";
            }
        }
        std::cout << std::endl;
    }
}*/

}

#endif