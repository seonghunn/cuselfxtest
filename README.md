# Self-intersection check with lbvh using GPU
- Parallel Mesh self-intersection test in GPU
- All algorithms including BVH construction and query check, triangle intersection test are run in parallel at GPU
- CUDA 11.8
## Run
```
chmod +x setup.sh
./setup.sh
cd build
make
./lbvh <mesh filename in current path>
```