#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#define STACK_SIZE 256
#include "predicator.cuh"

namespace lbvh
{
    // query object indices that potentially overlaps with query aabb.
    //
    // requirements:
    // - OutputIterator should be writable and its object_type should be uint32_t
    //

    // template <typename Real, typename Objects, bool IsConst, typename OutputIterator>
    // __device__ unsigned int query_device(
    //     const detail::basic_device_bvh<Real, Objects, IsConst> &bvh,
    //     const query_overlap<Real> q, OutputIterator outiter,
    //     const unsigned int query_idx,
    //     const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept
    // {
    //     using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
    //     using index_type = typename bvh_type::index_type;
    //     using aabb_type = typename bvh_type::aabb_type;
    //     using node_type = typename bvh_type::node_type;

    //     index_type stack[STACK_SIZE]; // is it okay?
    //     index_type *stack_ptr = stack;
    //     OutputIterator outiter_begin = outiter;
    //     *stack_ptr++ = 0; // root node is always 0
    //     // printf("Target idx : %d\n", query_idx);
    //     // printf("Target AABB: Lower[%f, %f, %f], Upper[%f, %f, %f]\n",
    //     //        q.target.lower.x, q.target.lower.y, q.target.lower.z,
    //     //        q.target.upper.x, q.target.upper.y, q.target.upper.z);

    // unsigned int num_found = 0;
    // int iter = 0;

    // do
    // {
    //     printf("iter %d\n",iter++);
    //     //for(int i = 0; i < STACK_SIZE; i++){
    //     for(int i = 0; i < 300; i++){
    //         //if(stack[i]==0) break;
    //         printf("%d ", stack[i]);
    //     }
    //     printf("\n");

    //     const index_type node = *--stack_ptr;
    //     const index_type L_idx = bvh.nodes[node].left_idx;
    //     const index_type R_idx = bvh.nodes[node].right_idx;

    //     // 오른쪽 자식 노드 먼저 확인
    //     if (intersects(q.target, bvh.aabbs[R_idx])) {
    //         *stack_ptr++ = R_idx;
    //         printf("stack_ptr %d\n",*stack_ptr);
    //     }

    //     // 왼쪽 자식 노드 확인
    //     if (intersects(q.target, bvh.aabbs[L_idx])) {
    //         *stack_ptr++ = L_idx;
    //         printf("stack_ptr %d\n",*stack_ptr);
    //     }

    //     // 현재 노드 처리
    //     const auto obj_idx = bvh.nodes[node].object_idx;
    //     if (obj_idx != 0xFFFFFFFF && obj_idx != query_idx) {
    //         if (num_found < max_buffer_size) {
    //             *outiter++ = obj_idx;
    //         }
    //         ++num_found;
    //     }
    // } while (stack < stack_ptr);

    // return num_found;
    // }
    template <typename Real, typename Objects, bool IsConst, typename OutputIterator>
    __device__ unsigned int get_number_of_intersect_candidates(
        const detail::basic_device_bvh<Real, Objects, IsConst> &bvh,
        const query_overlap<Real> q, OutputIterator outiter,
        const unsigned int query_idx,
        const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        index_type stack[STACK_SIZE]; // is it okay?
        index_type *stack_ptr = stack;
        OutputIterator outiter_begin = outiter;
        *stack_ptr++ = 0; // root node is always 0
        // printf("Target idx : %d\n", query_idx);
        // printf("Target AABB: Lower[%f, %f, %f], Upper[%f, %f, %f]\n",
        //        q.target.lower.x, q.target.lower.y, q.target.lower.z,
        //        q.target.upper.x, q.target.upper.y, q.target.upper.z);
        unsigned int num_found = 0;
        do
        {
            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.target, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                //if(obj_idx == query_idx) continue;
                // printf("Obj idx : %d , Left Node AABB: Lower[%f, %f, %f], Upper[%f, %f, %f]\n", obj_idx,
                //        bvh.aabbs[L_idx].lower.x, bvh.aabbs[L_idx].lower.y, bvh.aabbs[L_idx].lower.z,
                //        bvh.aabbs[L_idx].upper.x, bvh.aabbs[L_idx].upper.y, bvh.aabbs[L_idx].upper.z);
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                }
            }
            if (intersects(q.target, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                // printf("Obj idx : %d, Right Node AABB: Lower[%f, %f, %f], Upper[%f, %f, %f]\n", obj_idx,
                //        bvh.aabbs[R_idx].lower.x, bvh.aabbs[R_idx].lower.y, bvh.aabbs[R_idx].lower.z,
                //        bvh.aabbs[R_idx].upper.x, bvh.aabbs[R_idx].upper.y, bvh.aabbs[R_idx].upper.z);
                //if(obj_idx == query_idx) continue;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                }
            }

        } while (stack < stack_ptr);

        // cudaDeviceSynchronize();
        // for (auto iter = outiter_begin; iter != outiter; ++iter)
        //{
        //    printf("current queryidx : %d, outiter value: %d\n", query_idx, *iter);
        //}
        // printf("current queryidx : %d, num_found %d\n", query_idx, num_found);
        return num_found;
    }



    // bfs code ----------------------
    template <typename Real, typename Objects, bool IsConst, typename OutputIterator>
    __device__ unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst> &bvh,
        const query_overlap<Real> q, OutputIterator outiter,
        const unsigned int query_idx,
        const unsigned int first = 0xFFFFFFFF) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        index_type stack[STACK_SIZE]; // is it okay?
        index_type *stack_ptr = stack;
        OutputIterator outiter_begin = outiter;
        *stack_ptr++ = 0; // root node is always 0
        // printf("Target idx : %d\n", query_idx);
        // printf("Target AABB: Lower[%f, %f, %f], Upper[%f, %f, %f]\n",
        //        q.target.lower.x, q.target.lower.y, q.target.lower.z,
        //        q.target.upper.x, q.target.upper.y, q.target.upper.z);
        unsigned int num_found = 0;
        //printf("current queryidx : %d, num_found %d\n", query_idx, num_found);
        int iter = 0;

        // dynamic buffer
        outiter += first;

        //*outiter++ = query_idx;
        //if(query_idx == 0 || query_idx == 1 || query_idx == 767 || query_idx == 768) printf("query idx %d max buffer %d\n", query_idx, first);
        do
        {
            // printf("iter %d\n",iter++);
            // //for(int i = 0; i < STACK_SIZE; i++){
            // for(int i = 0; i < 200; i++){
            //     //if(stack[i]==0) break;
            //     printf("%d ", stack[i]);
            // }
            // printf("\n");
            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.target, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                //if(obj_idx == query_idx) continue;
                // printf("Obj idx : %d , Left Node AABB: Lower[%f, %f, %f], Upper[%f, %f, %f]\n", obj_idx,
                //        bvh.aabbs[L_idx].lower.x, bvh.aabbs[L_idx].lower.y, bvh.aabbs[L_idx].lower.z,
                //        bvh.aabbs[L_idx].upper.x, bvh.aabbs[L_idx].upper.y, bvh.aabbs[L_idx].upper.z);
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        //if (num_found < max_buffer_size)
                        //{
                            *outiter++ = query_idx;
                            *outiter++ = obj_idx;
                        //}
                        ++num_found;
                        //printf("add obj idx %d\n", obj_idx);
                    }
                    // else{
                    //     printf("obj idx %d query idx %d\n",obj_idx, query_idx);
                    // }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                    //printf("left interesect query idx %d stack_ptr %d Lidx %d\n",query_idx, *stack_ptr, L_idx);
                }
            }
            if (intersects(q.target, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                // printf("Obj idx : %d, Right Node AABB: Lower[%f, %f, %f], Upper[%f, %f, %f]\n", obj_idx,
                //        bvh.aabbs[R_idx].lower.x, bvh.aabbs[R_idx].lower.y, bvh.aabbs[R_idx].lower.z,
                //        bvh.aabbs[R_idx].upper.x, bvh.aabbs[R_idx].upper.y, bvh.aabbs[R_idx].upper.z);
                //if(obj_idx == query_idx) continue;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if(obj_idx != query_idx)
                    {
                        //if (num_found < max_buffer_size)
                        //{
                            *outiter++ = query_idx;
                            *outiter++ = obj_idx;
                        //}
                        ++num_found;
                        //printf("add obj idx %d\n", obj_idx);
                    }
                    // else{
                    //     printf("obj idx %d query idx %d\n",obj_idx, query_idx);
                    // }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                    //printf("right intersect query idx %d stack_ptr %d Ridx %d\n",query_idx, *stack_ptr, R_idx);
                }
            }
        //printf("iter %d, stack %d\n", iter-1, stack);

        } while (stack < stack_ptr);

        //cudaDeviceSynchronize();
        // for (auto iter = outiter_begin; iter != outiter; ++iter)
        // {
        //    printf("current queryidx : %d, outiter value: %d\n", query_idx, *iter);
        // }
        //printf("current queryidx : %d, num_found %d\n", query_idx, num_found);
        return num_found;
    }

}

#endif // LBVH_QUERY_CUH