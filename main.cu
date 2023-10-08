#include "selfx.cuh"
#include "include/morton_code.cuh"
#include <igl/read_triangle_mesh.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>
#include <time.h>

#include "tri_tri_intersect.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    //string INPUT_PATH = "/seoh_fast_cephfs/model/";
    string INPUT_PATH = "";
    string input_filename(argv[1]);
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // clock_t start_cpu, end_cpu;
    // start_cpu = clock();
    igl::read_triangle_mesh(INPUT_PATH + input_filename, V, F);

    const std::size_t N = F.rows();

    // Vertex and Faces
    Vertex<float>* V_array = new Vertex<float>[V.rows()];
    Face<int>* F_array = new Face<int>[F.rows()];

    // Copy data Eigen to array
    for (int i = 0; i < V.rows(); ++i)
    {
        V_array[i] = {
            static_cast<float>(V(i, 0)), 
            static_cast<float>(V(i, 1)), 
            static_cast<float>(V(i, 2))
        };
    }

    for (int i = 0; i < F.rows(); ++i)
    {
        F_array[i] = {F(i, 0), F(i, 1), F(i, 2)};
    }

    // V와 F로부터 삼각형(Triangle) 목록을 초기화합니다.
    /*
    for (int i = 0; i < F.rows(); i++)
    {
        Triangle tri;
        tri.v0 = make_float3(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2));
        tri.v1 = make_float3(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2));
        tri.v2 = make_float3(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2));
        triangles.push_back(tri);
    }*/

    unsigned int num_vertices = V.rows();
    unsigned int num_faces = F.rows();
    // thrust::device_vector<unsigned int> adj_faces_dev(num_faces * FACES_SIZE, 0xFFFFFFFF);
    // lbvh::adj_faces(V_vector, F_vector, adj_faces_dev, num_faces);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    bool isIntersect = selfx::self_intersect(V_array, F_array, num_vertices, num_faces);
    // bool tri_intersect = lbvh::tri_tri_intersect(triangles);

    // 1 : self-intersect 0 : self-intersection free
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total self-intersect test runtime in main.cu %f ms\n", milliseconds);
    std::cout << "self intersect test : " << (isIntersect ? "self-intersect" : "self-intersect-free") << std::endl;
    // std::cout<<"total runtime : "<<(double)(end_cpu - start_cpu)/ CLOCKS_PER_SEC<<" sec"<<std::endl;

    // lbvh::parallel_adj_faces(V, F);

    /*
        float3 a = make_float3(1,2,3);
        float3 b = make_float3(5,6,7);
        float3 sum = a + b;
        float3 sub = a - b;
        float3 crossRes = cross(a,b);
        float3 crossRes1 = cross(b,a);
        float dotRes = dot(a,b);

        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "sum: " << sum << std::endl;
        std::cout << "sub: " << sub << std::endl;
        std::cout << "cross(a,b): " << crossRes << std::endl;
        std::cout << "cross(b,a): " << crossRes1 << std::endl;
        std::cout << "dot(a,b): " << dotRes << std::endl;
        */

    /*
    float3 p1, p2, q1, q2, r1, r2, source, target;
    bool coplanar;
    p1 = triangles[0].v0;
    q1 = triangles[0].v1;
    r1 = triangles[0].v2;
    p2 = triangles[1].v0;
    q2 = triangles[1].v1;
    r2 = triangles[1].v2;
    bool testtest = lbvh::tri_tri_intersection_test_3d(p1,q1,r1,p2,q2,r2,coplanar,source,target);
    std::cout<<"triangle intersection test 3d : "<<testtest<<
    " "<<source<<" "<<target<<std::endl;
*/
    delete[] V_array;
    delete[] F_array;

    return 0;
}
