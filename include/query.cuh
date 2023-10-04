#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#define STACK_SIZE 1024

namespace lbvh
{
    // query object indices that potentially overlaps with query aabb.
    //
    // requirements:
    // - OutputIterator should be writable and its object_type should be uint32_t
    //

    template <typename Real, typename Objects, bool IsConst, typename OutputIterator>
    __device__ unsigned int query_device(
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
                        if (num_found < max_buffer_size)
                        {
                            *outiter++ = obj_idx;
                        }
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
                        if (num_found < max_buffer_size)
                        {
                            *outiter++ = obj_idx;
                        }
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

}

#endif // LBVH_QUERY_CUH