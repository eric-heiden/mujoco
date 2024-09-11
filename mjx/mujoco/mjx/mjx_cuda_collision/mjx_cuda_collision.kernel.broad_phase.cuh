namespace mujoco
{
    namespace mjx
    {
        namespace cuda
        {
            // ====================================================================
             // inilne functions 
             // ====================================================================

             __device__ __forceinline__ bool intersectAABB(const float* __restrict d_bvh_aabb_dyn, const unsigned int i, const unsigned int k)
            {
                float3 pos_i = make_float3(d_bvh_aabb_dyn[i * 6 + 0], d_bvh_aabb_dyn[i * 6 + 1], d_bvh_aabb_dyn[i * 6 + 2]);
                float3 pos_k = make_float3(d_bvh_aabb_dyn[k * 6 + 0], d_bvh_aabb_dyn[k * 6 + 1], d_bvh_aabb_dyn[k * 6 + 2]);
                float3 size_i = make_float3(d_bvh_aabb_dyn[i * 6 + 3], d_bvh_aabb_dyn[i * 6 + 4], d_bvh_aabb_dyn[i * 6 + 5]);
                float3 size_k = make_float3(d_bvh_aabb_dyn[k * 6 + 3], d_bvh_aabb_dyn[k * 6 + 4], d_bvh_aabb_dyn[k * 6 + 5]);
                float3 min_i = make_float3(pos_i.x - size_i.x, pos_i.y - size_i.y, pos_i.z - size_i.z);
                float3 max_i = make_float3(pos_i.x + size_i.x, pos_i.y + size_i.y, pos_i.z + size_i.z);
                float3 min_k = make_float3(pos_k.x - size_k.x, pos_k.y - size_k.y, pos_k.z - size_k.z);
                float3 max_k = make_float3(pos_k.x + size_k.x, pos_k.y + size_k.y, pos_k.z + size_k.z);

                if (max_i.x < min_k.x) { return false; }
                if (max_i.y < min_k.y) { return false; }
                if (max_i.z < min_k.z) { return false; }
                if (min_i.x > max_k.x) { return false; }
                if (min_i.y > max_k.y) { return false; }
                if (min_i.z > max_k.z) { return false; }
                return true;
            }


            __device__ __forceinline__ unsigned int mapCoord_NxN(
                const unsigned int gi, const unsigned int m_ngeom)
            {
                if (gi >= m_ngeom * m_ngeom) { return 0xFFFFFFFF; } // outside of the map 
                unsigned int i = gi / m_ngeom;
                unsigned int k = gi % m_ngeom;
                if (i < k) { return (i * m_ngeom + k); }    // only runs upper diagonal area 
                return 0xFFFFFFFF;
            }

            // a and b: the array range of ix
            // v: the criterion
            __device__ __forceinline__ unsigned int bisection(const unsigned int* __restrict ix, const unsigned int iv, unsigned int a, unsigned int b)
            {
                unsigned int c = 0;
                while (b - a > 1)
                {
                    c = (a + b) / 2;
                    unsigned int fc = ix[c];
                    if (fc <= iv) { a = c; }
                    else { b = c; }
                }
                c = a;
                if (c != b && (ix[b] <= iv)) { c = b; }
                return c;
            }


            // ====================================================================
            // space AABB
            // ====================================================================
            __global__ void initSpaceAABB(int* __restrict d_spaceAABB)
            {
                unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;
                if (gi == 0)
                {
                    d_spaceAABB[0] = 0x7FFFFFFF;
                    d_spaceAABB[1] = -0x7FFFFFFF;
                }
            }

            __global__ void calculateSpaceAABB(const unsigned int m_ngeom,
                int* __restrict d_spaceAABB,
                const float* __restrict d_bvh_aabb_dyn, const float size_limit, const float precisionInv, const int axis)
            {
                unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;
                if (gi < m_ngeom)
                {
                    //  mjtNum*  d_bvh_aabb_dyn;     // global bounding box (center, size)               (nbvhdynamic x 6)
                    float aabb_center = d_bvh_aabb_dyn[gi * 6 + axis];
                    float aabb_size = d_bvh_aabb_dyn[gi * 6 + 3 + axis];
                    if(aabb_size > size_limit){ return; } // ignore if the object is too big (e.g. plane)
                    int aabb_min_i = (int) floor((aabb_center - aabb_size) * precisionInv);
                    int aabb_max_i = (int) ceil((aabb_center + aabb_size) * precisionInv);

                    atomicMin(&d_spaceAABB[0], aabb_min_i);
                    atomicMax(&d_spaceAABB[1], aabb_max_i);
                }
            }

            // ====================================================================
            // Geom BV: transform local AABBs to be global AABBs
            // ====================================================================
            __global__ void calculateAABB(const unsigned int m_ngeom,
                const float* __restrict m_aabb, const float* __restrict d_xpos, const float* __restrict d_xmat,
                float* __restrict d_bvh_aabb_dyn)
            {
                unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;
                if (gi < m_ngeom)
                {
                    const float aabb[3] = {m_aabb[gi * 6 + 3], m_aabb[gi * 6 + 4], m_aabb[gi * 6 + 5]}; //halfSize

                    const float3 corner[8] = {
                        make_float3(-aabb[0], -aabb[1], -aabb[2]),
                        make_float3(aabb[0], -aabb[1], -aabb[2]),
                        make_float3(-aabb[0],  aabb[1], -aabb[2]),
                        make_float3(aabb[0],  aabb[1], -aabb[2]),
                        make_float3(-aabb[0], -aabb[1],  aabb[2]),
                        make_float3(aabb[0], -aabb[1],  aabb[2]),
                        make_float3(-aabb[0],  aabb[1],  aabb[2]),
                        make_float3(aabb[0],  aabb[1],  aabb[2])};

                    const float pos[3] = {d_xpos[gi * 3 + 0], d_xpos[gi * 3 + 1], d_xpos[gi * 3 + 2]};

                    const float r[9] = {
                        d_xmat[gi * 9 + 0],d_xmat[gi * 9 + 1],d_xmat[gi * 9 + 2],
                        d_xmat[gi * 9 + 3],d_xmat[gi * 9 + 4],d_xmat[gi * 9 + 5],
                        d_xmat[gi * 9 + 6],d_xmat[gi * 9 + 7],d_xmat[gi * 9 + 8]
                    };

                    float aabb_max[3] = {(float) -0x7FFFFFFF, (float) -0x7FFFFFFF, (float) -0x7FFFFFFF};
                    // aabb_max = -aabb_min

                    for (int k = 0; k < 8; k++)
                    {
                        float np_x = r[0] * corner[k].x + r[1] * corner[k].y + r[2] * corner[k].z;
                        float np_y = r[3] * corner[k].x + r[4] * corner[k].y + r[5] * corner[k].z;
                        float np_z = r[6] * corner[k].x + r[7] * corner[k].y + r[8] * corner[k].z;

                        aabb_max[0] = (aabb_max[0] > np_x) ? aabb_max[0] : np_x;
                        aabb_max[1] = (aabb_max[1] > np_y) ? aabb_max[1] : np_y;
                        aabb_max[2] = (aabb_max[2] > np_z) ? aabb_max[2] : np_z;
                    }

                    d_bvh_aabb_dyn[gi * 6 + 0] = pos[0];
                    d_bvh_aabb_dyn[gi * 6 + 1] = pos[1];
                    d_bvh_aabb_dyn[gi * 6 + 2] = pos[2];
                    d_bvh_aabb_dyn[gi * 6 + 3] = aabb_max[0];
                    d_bvh_aabb_dyn[gi * 6 + 4] = aabb_max[1];
                    d_bvh_aabb_dyn[gi * 6 + 5] = aabb_max[2];
                }
            }

            __global__ void quantizeAABB(const unsigned int m_ngeom,
                unsigned int* __restrict d_aabb_min_quant, unsigned int* __restrict d_aabb_max_quant, unsigned int* __restrict d_geom_idx,
                const float* __restrict d_bvh_aabb_dyn,
                const int axis, const float spaceMin, const float invSpaceWidth, const int precision, const char axisFlip)
            {
                unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;
                if (gi < m_ngeom)
                {
                    float aabb_center = d_bvh_aabb_dyn[gi * 6 + axis];
                    float aabb_size = d_bvh_aabb_dyn[gi * 6 + 3 + axis];

                    float min = ((aabb_center - aabb_size - spaceMin) * invSpaceWidth * (float) precision);
                    float max = ((aabb_center + aabb_size - spaceMin) * invSpaceWidth * (float) precision);
                    int min_i = (int)min;
                    int max_i = (int)max;
                    min_i = (min_i < 0) ? 0 : (min_i > precision - 1) ? (precision - 1) : min_i;
                    max_i = (max_i < 0) ? 0 : (max_i > precision - 1) ? (precision - 1) : max_i;

                    d_aabb_min_quant[gi] = (!axisFlip) ? min_i : (precision - max_i);
                    d_aabb_max_quant[gi] = (!axisFlip) ? max_i : (precision - min_i);

                    d_geom_idx[gi] = gi;
                }
            }


            // ====================================================================
            // NxN
            // ====================================================================
            __global__ void candidateNxN(const unsigned int m_ngeom,
                unsigned int* __restrict col_counter,
                unsigned int* __restrict col_pair,
                const float* __restrict d_bvh_aabb_dyn)
            {
                unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;    // coordinate in full rank matrix 
                unsigned int mapCoord = mapCoord_NxN(gi, m_ngeom);
                if (mapCoord == 0xFFFFFFFF) { return; }
                unsigned int i = mapCoord / m_ngeom, k = mapCoord % m_ngeom;

                if (intersectAABB(d_bvh_aabb_dyn, i, k))
                {
                    unsigned int pair_idx = atomicAdd(&col_counter[0], 1);
                    col_pair[pair_idx * 2] = (i < k) ? i : k;
                    col_pair[pair_idx * 2 + 1] = (i < k) ? k : i;
                }
            }



            // ====================================================================
            // sort axis 
            // ====================================================================
            __global__ void sortAABBMax(const unsigned int m_ngeom,
                unsigned int* __restrict d_aabb_max_quant_sorted,
                const unsigned int* __restrict d_aabb_max_quant, const unsigned int* __restrict geom_idx_sorted)
            {
                unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;
                if (gi < m_ngeom)
                {
                    int sorted_idx = geom_idx_sorted[gi];
                    d_aabb_max_quant_sorted[gi] = d_aabb_max_quant[sorted_idx];
                }
            }


            __global__ void countCollidingAABB(const unsigned int m_ngeom,
                unsigned int* __restrict d_col_count, unsigned int* __restrict d_aabb_min_quant, unsigned int* __restrict d_aabb_max_quant)
            {
                unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;
                if (gi < m_ngeom)
                {
                    unsigned int st_max = d_aabb_max_quant[gi];
                    unsigned int pi = bisection(d_aabb_min_quant, st_max, 0, m_ngeom - 1);
                    d_col_count[gi] = (pi - gi);
                    if (gi == 0) { d_col_count[m_ngeom] = 0; } // the last cell for sorting 
                }
            }

            __global__ void testAABBPerPair(const unsigned int m_npair, const unsigned int m_ngeom,
                unsigned int* __restrict d_col_counter_sort,
                unsigned int* __restrict d_col_pair_sort,
                const unsigned int* __restrict d_col_offset, const unsigned int* __restrict d_geom_idx, const float* __restrict d_bvh_aabb_dyn)
            {
                unsigned int pi = blockIdx.x * blockDim.x + threadIdx.x;
                if (pi < m_npair)
                {
                    unsigned int gi = bisection(d_col_offset, pi, 0, m_ngeom - 1);
                    unsigned int offset = d_col_offset[gi];
                    unsigned int gk = gi + (pi - offset) + 1;
                    unsigned int i = d_geom_idx[gi], k = d_geom_idx[gk];
                    if (i == k) { return; } // sorting implies that indexes are sorted 

                    if (intersectAABB(d_bvh_aabb_dyn, i, k))
                    {
                        unsigned int pair_idx = atomicAdd(&d_col_counter_sort[0], 1);
                        d_col_pair_sort[pair_idx * 2] = (i < k) ? i : k;
                        d_col_pair_sort[pair_idx * 2 + 1] = (i < k) ? k : i;
                    }
                }
            }
        }
    }
}
