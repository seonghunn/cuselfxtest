# Self-intersection check with lbvh using GPU
- Parallel Mesh self-intersection test in GPU
- All algorithms including BVH construction and query check, triangle intersection test are run in parallel at GPU
## Run
```
cd cuselfxtest
pip install -e .
cd build
./cuselfxtest <mesh filepath>
```