#include "cuselfxtest/selfx.cuh"
//#include "include/morton_code.cuh"
#include <igl/read_triangle_mesh.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>
#include <time.h>

//#include "tri_tri_intersect.cuh"

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
    Triangle<int>* F_array = new Triangle<int>[F.rows()];

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

    unsigned int num_vertices = V.rows();
    unsigned int num_faces = F.rows();
    bool isIntersect = selfx::self_intersect(V_array, F_array, num_vertices, num_faces);
    std::cout << "self intersect test : " << (isIntersect ? "self-intersect" : "self-intersect-free") << std::endl;

    delete[] V_array;
    delete[] F_array;

    return 0;
}
