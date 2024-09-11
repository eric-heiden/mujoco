#include "mjx_cuda_collision.h"
#include "mjx_cuda_collision.common.h"

#include "helper_math.h"
#include "mjx_cuda_collision.kernel.broad_phase.cuh"
#include "mjx_cuda_collision.kernel.gjk.epa.cuh"

// thrust for radix_sort
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstring>

namespace mujoco
{
    namespace mjx
    {
        namespace cuda
        {
            /* in mjx/_src/types.py
            class GeomType(enum.IntEnum):
            """Type of geometry.

            Members:
                PLANE: plane
                HFIELD: height field
                SPHERE: sphere
                CAPSULE: capsule
                ELLIPSOID: ellipsoid
                CYLINDER: cylinder
                BOX: box
                MESH: mesh
                SDF: signed distance field
            """

            PLANE = mujoco.mjtGeom.mjGEOM_PLANE
            HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD
            SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
            CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
            ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
            BOX = mujoco.mjtGeom.mjGEOM_BOX
            MESH = mujoco.mjtGeom.mjGEOM_MESH
            # unsupported: NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE
            */

            static const int 
                mjGEOM_PLANE = 0, 
                mjGEOM_HFIELD = 1, 
                mjGEOM_SPHERE = 2, 
                mjGEOM_CAPSULE = 3, 
                mjGEOM_ELLIPSOID = 4, 
                mjGEOM_CYLINDER = 5, 
                mjGEOM_BOX = 6, 
                mjGEOM_MESH = 7, 
                mjGEOM_size = 8;


            struct gjk_input
            {
                // from Model 
                unsigned int m_ngeom = 0;                       // m.ngeom 
                jax_array 
                    * m_geom_size = NULL,                       // m.geom_size
                    * m_geom_dataid = NULL;                     // m.geom_dataids
                
                // from Data 
                jax_array
                    * d_geom_xpos = NULL,                       // d_geom_xpos
                    * d_geom_xmat = NULL;                       // d.geom_xmat
                
                // broad-phase collision
                unsigned int                    
                    candidate_pair_count_max = 0,               // contact.geom.shape[0]: Number of d_candidate_pairs
                    candidate_pair_count = 0;                   // Actual active candidate pairs out of candidate_pair_count_max
                jax_array 
                    * d_candidate_pairs = NULL;                 // broad-phase result. shape = (candidate_pair_count, 2)

                // merged convex mesh 
                uint2 key_types = make_uint2(0, 0);             // FunctionKey.types
                jax_array 
                    * m_mesh_convex_vert = NULL,                // Merged m.mesh_convex[].vert shape = (sum(len(m.mesh_vertex[])), 3)
                    * m_mesh_convex_vert_offset = NULL;         // Offset of m_mesh_convex_vert[] sizes, shape = (len(m.mesh_vertex) + 1)

                // narrow-phase collision: gjk 
                unsigned int ncon = 0;                          // _COLLISION_FUNC[].ncon or manually set to (1, 4) for GJK

                // gjk parameters 
                float depthExtension = 0.0f;
                int gjkIterationCount= 0, epaIterationCount= 0, epaBestCount= 0, multiPolygonCount = 0;
                float multiTiltAngle = 0.0f;

                bool compress_result = false;
            };
            
            // [gjk_output]
            //   - data*: Optional, updates if the pointer is not null
            // if (compress_result):
            //   1. counter                  : (1, 1)       : Number of active contacts (c)
            //   2. pos, dist, frame, normal*: (c, dim)     : Packed data of active contacts only
            //   3. pairs*                   : (c, 2)       : Pairs of active contacts
            // else:
            //   1. counter                  : (candidate_pair_count, 1)   : 1 if contact.geom[i] is active
            //   2. pos, dist, frame, normal*: (candidate_pair_count, dim) : i-th data updated if (counter[i] == 1)
            //                                                               dist[i] = gjk_contact_distance_invalid if (counter[i] == 0)
            //   3. pairs*                   : (candidate_pair_count, 2)   : Same as d_candidate_pairs

            struct gjk_output
            {
                jax_array
                *d_contact_counter = NULL, 
                *d_contact_pos = NULL,      
                *d_contact_dist = NULL,     
                *d_contact_frame = NULL,
                *d_contact_normal = NULL, 
                *d_contact_simplex = NULL, 
                *d_contact_pairs = NULL;    // *Optional, packed pairs of active contacts
            };

            bool calculate_bvh_aabb_dyn(cudaStream_t stream,
                const int m_ngeom, jax_array& m_aabb, jax_array& d_geom_xpos, jax_array& d_geom_xmat,
                jax_array& d_bvh_aabb_dyn)
            {
                _ASSERT_(assert_array_alloc(m_aabb, sizeof(float) * 6 * m_ngeom, "m_aabb"));
                _ASSERT_(assert_array_alloc(d_geom_xpos, sizeof(float) * 3 * m_ngeom, "d_geom_xpos"));
                _ASSERT_(assert_array_alloc(d_geom_xmat, sizeof(float) * 9 * m_ngeom, "d_geom_xmat"));
                _ASSERT_(assert_array_alloc(d_bvh_aabb_dyn, sizeof(float) * 6 * m_ngeom, "d_bvh_aabb_dyn"));

                {
                    const unsigned int blockSize = 256;
                    const unsigned int gridSize = (m_ngeom + blockSize - 1) / blockSize;
                    calculateAABB << <gridSize, blockSize, 0, stream >> > (m_ngeom,
                        m_aabb.dev.f, d_geom_xpos.dev.f, d_geom_xmat.dev.f, d_bvh_aabb_dyn.dev.f);
                    _ASSERT_(assert_cuda("calculateAABB"));
                }

                return true;
            }

            // N x N
            // filter: For each geom_pair, build a filter for the pair.
            bool collision_candidate_NxN(cudaStream_t stream,
                const unsigned int m_ngeom, const unsigned int max_geom_pairs, 
                jax_array& d_bvh_aabb_dyn, jax_array& col_counter_nxn, jax_array& col_pair_nxn)
            {
                _ASSERT_(assert_array_alloc(d_bvh_aabb_dyn, sizeof(unsigned int) * m_ngeom * 6, "d_bvh_aabb_dyn"));
                _ASSERT_(assert_array_alloc(col_counter_nxn, sizeof(unsigned int), "col_counter_nxn"));
                _ASSERT_(assert_array_alloc(col_pair_nxn, sizeof(unsigned int) *max_geom_pairs * 2, "col_pair_nxn"));

                // counter = col_pair_nxn[0] = 0
                array_zero(col_counter_nxn, sizeof(unsigned int));

                // If a pair is given, run according to the list; otherwise, it is N x N.
                {
                    // N x N -> N x m_geom; computation times change, e.g., 544 -> 422, 789 -> 145, 26 -> 23 ms
                    const unsigned int blockSize = 256;
                    const unsigned int gridSize = ((m_ngeom * m_ngeom) + blockSize - 1) / blockSize;
                    candidateNxN << <gridSize, blockSize, 0, stream >> > (m_ngeom,
                        col_counter_nxn.dev.ui, col_pair_nxn.dev.ui, d_bvh_aabb_dyn.dev.f);
                    _ASSERT_(assert_cuda("kernel::candidateNxN"));
                }

                return true;
            }

            bool update_spaceAABB(cudaStream_t stream,
                float& spaceMin_axis, float& spaceMax_axis,
                jax_array* spaceAABB,
                const int axis, const int m_ngeom, jax_array& d_bvh_aabb_dyn, const float spaceAABB_sizeLimit, const float spaceAABB_precision)
            {
                _ASSERT_(assert_array_alloc(d_bvh_aabb_dyn, sizeof(float) * 6 * m_ngeom, "d_bvh_aabb_dyn"));
                _ASSERT_(assert_array_alloc(*spaceAABB, sizeof(int) * 2, "spaceAABB"));

                // aabb[0] = MAX, aabb[1] = -MAX
                {
                    initSpaceAABB << <1, 1, 0, stream >> > (spaceAABB->dev.i);
                    _ASSERT_(assert_cuda("initSpaceAABB"));
                }

                // aabb[0] = min_i(bv_aabb_dyn[i*3 + axis]), aabb[1] = max_i(bv_aabb_dyn[i*3+axis])
                {
                    const unsigned int blockSize = 256;
                    const unsigned int gridSize = (m_ngeom + blockSize - 1) / blockSize;
                    calculateSpaceAABB << <gridSize, blockSize, 0, stream >> > (m_ngeom,
                        spaceAABB->dev.i, d_bvh_aabb_dyn.dev.f, spaceAABB_sizeLimit, 1.0f / spaceAABB_precision, axis);
                    _ASSERT_(assert_cuda("kernel::calculateSpaceAABB"));
                }

                // aabb -> cpu 
                {
                    _ASSERT_(array_copy_from_device(*spaceAABB, sizeof(int) * 2));
                    int spaceMin_axis_i = 0, spaceMax_axis_i = 0;
                    spaceMin_axis_i = spaceAABB->cpu.i[0];
                    spaceMax_axis_i = spaceAABB->cpu.i[1];
                    spaceMin_axis = (float) spaceMin_axis_i * spaceAABB_precision;
                    spaceMax_axis = (float) spaceMax_axis_i * spaceAABB_precision;

                    // Extend the size of AABB to ensure that all AABBs are within the space
                    spaceMin_axis -= spaceAABB_precision * 5.0f;
                    spaceMax_axis += spaceAABB_precision * 5.0f;
                }

                return true;
            }

            // So that it is a multiple of 16
            inline int get_buffer_size(const int size)
            {
                return (((size + 15) / 16) * 16);
            }

            bool collision_candidate_by_sort(cudaStream_t stream,
                const unsigned int m_ngeom,const unsigned int max_geom_pairs, jax_array& d_bvh_aabb_dyn, const int axis,            
                const float& spaceMin_axis, const float& spaceMax_axis,
                jax_array& col_counter_sort, jax_array& col_pair_sort,                                            
                jax_array& buffer)         
            {
                int buffer_size = get_buffer_size((m_ngeom + 1));
                _ASSERT_(assert_array_alloc(buffer, sizeof(unsigned int) * buffer_size * 4, "buffer"));

                jax_array buf0; buf0.dev.sizeInBytes = sizeof(unsigned int) * buffer_size; buf0.dev.ui = buffer.dev.ui; 
                jax_array buf1; buf1.dev.sizeInBytes = sizeof(unsigned int) * buffer_size; buf1.dev.ui = buffer.dev.ui + buffer_size; 
                jax_array buf2; buf2.dev.sizeInBytes = sizeof(unsigned int) * buffer_size; buf2.dev.ui = buffer.dev.ui + buffer_size * 2; 
                jax_array buf3; buf3.dev.sizeInBytes = sizeof(unsigned int) * buffer_size; buf3.dev.ui = buffer.dev.ui + buffer_size * 3; 

                // [1] aabb min, max to integer values 
                jax_array* aabb_min_quant = NULL;
                jax_array* aabb_max_quant = NULL;
                jax_array* geom_idx = NULL;
                {
                    aabb_min_quant = &buf0;
                    aabb_max_quant = &buf1;
                    geom_idx = &buf2;

                    _ASSERT_(assert_array_alloc(*aabb_min_quant, sizeof(unsigned int) * m_ngeom, "array::aabb_min_quant"));
                    _ASSERT_(assert_array_alloc(*aabb_max_quant, sizeof(unsigned int) * m_ngeom, "array::aabb_max_quant"));
                    _ASSERT_(assert_array_alloc(*geom_idx, sizeof(unsigned int) * m_ngeom, "array::geom_idx"));

                    const int blockSize = 256;
                    const int gridSize = (m_ngeom + blockSize - 1) / blockSize;

                    quantizeAABB << <gridSize, blockSize, 0, stream >> > (m_ngeom,
                        aabb_min_quant->dev.ui, aabb_max_quant->dev.ui, geom_idx->dev.ui,
                        d_bvh_aabb_dyn.dev.f,
                        axis, spaceMin_axis, 1.0f / (spaceMax_axis - spaceMin_axis), (int) 0x7FFFFFF0, (char) 0);
                    _ASSERT_(assert_cuda("kernel::quantizeAABB"));
                }
                // buf[aabb_min_quant*, aabb_max_quant*, geom_idx*, -]


                // [2] sort (min, index) by key (min)
                jax_array* aabb_min_quant_sorted = aabb_min_quant;
                jax_array* geom_idx_sorted = geom_idx;
                {
                    thrust::device_ptr<unsigned int> dev_key(aabb_min_quant->dev.ui);
                    thrust::device_ptr<unsigned int> dev_value(geom_idx->dev.ui);
                    thrust::sort_by_key(thrust::device, dev_key, dev_key + m_ngeom, dev_value);
                }
                // buf[aabb_min_quant_sorted*, aabb_max_quant, geom_idx_sorted*, -]

                // [3] sort (max) by key (min)
                jax_array* aabb_max_quant_sorted = NULL;
                {
                    aabb_max_quant_sorted = &buf3;
                    _ASSERT_(assert_array_alloc(*aabb_max_quant_sorted, sizeof(unsigned int) * m_ngeom, "array::aabb_max_quant_sorted"));

                   // Sort max data according to the sorted min data
                    const int blockSize = 256;
                    const int gridSize = (m_ngeom + blockSize - 1) / blockSize;
                    sortAABBMax << <gridSize, blockSize, 0, stream >> > (m_ngeom,
                        aabb_max_quant_sorted->dev.ui, aabb_max_quant->dev.ui, geom_idx_sorted->dev.ui);
                    _ASSERT_(assert_cuda("kernel::sortAABBMax"));
                }
                // buf[aabb_min_quant_sorted, aabb_max_quant, geom_idx_sorted, aabb_max_quant_sorted*]

                // [4] Count the collision candidates per geom in the axis
                jax_array* col_candidate_count = NULL;
                {
                    col_candidate_count = &buf1;
                    _ASSERT_(assert_array_alloc(*col_candidate_count, sizeof(unsigned int) * (m_ngeom + 1), "array::col_candidate_count"));

                    const int blockSize = 256;
                    const int gridSize = (m_ngeom + blockSize - 1) / blockSize;
                    countCollidingAABB << <gridSize, blockSize, 0, stream >> > (m_ngeom,
                        col_candidate_count->dev.ui, aabb_min_quant_sorted->dev.ui, aabb_max_quant_sorted->dev.ui);
                    _ASSERT_(assert_cuda("kernel::countCollidingAABB"));
                }
                // buf[aabb_min_quant_sorted, col_candidate_count*, geom_idx_sorted, aabb_max_quant_sorted]

                // [5] Total number of collision candidates
                unsigned int total_pair_candidate = 0;
                {
                    cudaDeviceSynchronize(); 
                    unsigned int* x = col_candidate_count->dev.ui;
                    thrust::exclusive_scan(thrust::device, x, x + (m_ngeom + 1), x, 0);
                    _ASSERT_(array_get_at(total_pair_candidate, *col_candidate_count, m_ngeom));
                }

                // [6-1] No collision candidates -> No collision
                if (total_pair_candidate == 0)
                {
                    // col_pair_sort[0] = 0
                    _ASSERT_(array_zero(col_counter_sort, sizeof(unsigned int)));
                    return true;
                }
                // [6-2] Run per-candidate AABB checks
                else
                {
                    _ASSERT_(assert_array_alloc(col_pair_sort, sizeof(unsigned int) * (max_geom_pairs * 2), "array::col_pair_sort"));

                    {
                        // This version uses atomic operations but does not include sorting or packing of indexes (ui2).
                        _ASSERT_(array_zero(col_counter_sort, sizeof(unsigned int)));

                        const int blockSize = 256;
                        const int gridSize = (total_pair_candidate + blockSize - 1) / blockSize;
                        testAABBPerPair << <gridSize, blockSize, 0, stream >> > (total_pair_candidate, m_ngeom,
                            col_counter_sort.dev.ui, col_pair_sort.dev.ui,
                            col_candidate_count->dev.ui, geom_idx_sorted->dev.ui,
                            d_bvh_aabb_dyn.dev.f);
                        _ASSERT_(assert_cuda("kernel::testAABBPerPair"));
                    }
                }
                
                return true;
            }


            template <class GeomType1, class GeomType2> 
            bool _gjk_epa(cudaStream_t stream, gjk_output& output, gjk_input& input)
            {
                if (input.candidate_pair_count == 0) { return true; }
                _ASSERT_(input.m_geom_size);
                _ASSERT_(input.d_geom_xpos);
                _ASSERT_(input.d_geom_xmat);
                _ASSERT_(input.d_candidate_pairs);
                _ASSERT_(assert_array_alloc(*input.m_geom_size, sizeof(float) * input.m_ngeom * 3, "_gjk_epa::m_geom_size"));
                _ASSERT_(assert_array_alloc(*input.d_geom_xpos, sizeof(float) * input.m_ngeom * 3, "_gjk_epa::d_geom_xpos"));
                _ASSERT_(assert_array_alloc(*input.d_geom_xmat, sizeof(float) * input.m_ngeom * 9, "_gjk_epa::d_geom_xmat"));
                _ASSERT_(assert_array_alloc(*input.d_candidate_pairs, sizeof(uint) * 2 * input.candidate_pair_count, "_gjk_epa::d_candidate_pairs"));

                // ------------------------------------------------------------------------------------
                const bool compress_result = input.compress_result;
                // ------------------------------------------------------------------------------------
                // If compress_result == true:
                //   n = d_contact_counter[0] = total number of contacts
                //   d_contact_dist, pos, frame, etc[0, 1, 2, 3, 4, n - 1] = values
                // If compress_result == false:
                //   m = candidate_pair_count
                //   d_contact_count[0, 1, 2, 3, ..., m - 1] = 1 or 0
                //   d_contact_dist, pos, frame, etc[0, 1, 2, m - 1] = values if (d_contact_count[i] == 1)
                // ------------------------------------------------------------------------------------

                // d_contact_counter
                {
                    _ASSERT_(output.d_contact_counter);
                    _ASSERT_(assert_array_alloc(*output.d_contact_counter, 
                        sizeof(unsigned int) * (compress_result ? 1 : input.candidate_pair_count_max), "_gjk_epa::d_contact_counter"));
                }

                // pos & dist & normal & simplex & frame 
                {
                    _ASSERT_(output.d_contact_pos);
                    _ASSERT_(output.d_contact_dist);
                    _ASSERT_(output.d_contact_normal);
                    _ASSERT_(output.d_contact_simplex);
                    _ASSERT_(output.d_contact_frame);
                    _ASSERT_(assert_array_alloc(*output.d_contact_pos, sizeof(float) * 3 * input.ncon * input.candidate_pair_count_max, "_gjk_epa::d_contact_pos"));
                    _ASSERT_(assert_array_alloc(*output.d_contact_dist, sizeof(float) * input.ncon * input.candidate_pair_count_max, "_gjk_epa::d_contact_dist"));
                    _ASSERT_(assert_array_alloc(*output.d_contact_normal, sizeof(float) * 3 * input.candidate_pair_count_max, "_gjk_epa::d_contact_normal"));
                    _ASSERT_(assert_array_alloc(*output.d_contact_simplex, sizeof(float) * 12 * input.candidate_pair_count_max, "_gjk_epa::d_contact_simplex"));
                    _ASSERT_(assert_array_alloc(*output.d_contact_frame, sizeof(float) * 9 *  input.ncon * input.candidate_pair_count_max, "_gjk_epa::d_contact_frame"));
                }
                
                // Optional output. If the pointer is allocated, its size should be greater than the required size
                if(compress_result)
                {
                    _ASSERT_(output.d_contact_pairs);
                    _ASSERT_(assert_array_alloc(*output.d_contact_pairs, sizeof(int) * 2 * input.candidate_pair_count_max, "_gjk_epa::d_contact_pairs"));
                }
               
                // d_contact_dist = GK_CONTACT_DIST_INVALID for (candidate_pair_count_"max")
                if(!compress_result) 
                {
                    const int blockSize = 256;
                    const int gridSize = (input.candidate_pair_count_max + blockSize - 1) / blockSize;
                    GJK_EPA_InitContact<< <gridSize, blockSize, 0, stream >> > (
                        input.candidate_pair_count_max, input.ncon, 
                        output.d_contact_dist->dev.f);
                    assert_cuda("kernel::GJK_EPA_InitContact");
                }

                // run gjk
                {
                    const int blockSize = 256;
                    const int gridSize = (input.candidate_pair_count + blockSize - 1) / blockSize;
                    GJK<GeomType1, GeomType2> << <gridSize, blockSize, 0, stream >> > (
                        input.candidate_pair_count,
                        output.d_contact_normal->dev.f, output.d_contact_simplex->dev.f4,
                        input.d_candidate_pairs->dev.i, 
                        input.ncon, input.m_ngeom, input.m_geom_size->dev.f, input.d_geom_xpos->dev.f, input.d_geom_xmat->dev.f,
                        input.m_geom_dataid->dev.i, input.m_mesh_convex_vert->dev.f, input.m_mesh_convex_vert_offset->dev.i,
                        input.gjkIterationCount);
                    assert_cuda("kernel::GJK");
                }

                // run epa
                {
                    const int blockSize = 256;
                    const int gridSize = (input.candidate_pair_count + blockSize - 1) / blockSize;
                    EPA<GeomType1, GeomType2> << <gridSize, blockSize, 0, stream >> > (
                        input.candidate_pair_count,
                        output.d_contact_counter->dev.ui,
                        output.d_contact_dist->dev.f, output.d_contact_frame->dev.f, output.d_contact_normal->dev.f, output.d_contact_pairs->dev.i,
                        output.d_contact_simplex->dev.f4,
                        input.d_candidate_pairs->dev.i, 
                        input.ncon, input.m_ngeom, input.m_geom_size->dev.f, input.d_geom_xpos->dev.f, input.d_geom_xmat->dev.f,
                        input.m_geom_dataid->dev.i, input.m_mesh_convex_vert->dev.f, input.m_mesh_convex_vert_offset->dev.i,
                        input.depthExtension, input.epaIterationCount, input.epaBestCount, compress_result);
                    assert_cuda("kernel::EPA");
                }

                // Run multiple contacts if the new contact pair size > 0
                unsigned int contact_pair_count = input.candidate_pair_count;
                if(compress_result)
                {
                    _ASSERT_(array_get_at(contact_pair_count, *output.d_contact_counter, 0));
                }

                if(contact_pair_count > 0)
                {
                    const int blockSize = 256;
                    const int gridSize = (contact_pair_count+ blockSize - 1) / blockSize;
                    GetMultipleContacts<GeomType1, GeomType2> << <gridSize, blockSize, 0, stream >> > (
                        contact_pair_count,
                        output.d_contact_pos->dev.f,
                        output.d_contact_dist->dev.f, output.d_contact_normal->dev.f,
                        compress_result ? output.d_contact_pairs->dev.i : input.d_candidate_pairs->dev.i,
                        input.ncon, input.m_ngeom, input.m_geom_size->dev.f, input.d_geom_xpos->dev.f, input.d_geom_xmat->dev.f,
                        input.m_geom_dataid->dev.i, input.m_mesh_convex_vert->dev.f, input.m_mesh_convex_vert_offset->dev.i,
                        input.depthExtension, input.multiPolygonCount, input.multiTiltAngle);
                    assert_cuda("kernel::GetMultipleContacts");
                }

                return true;
            }

            bool gjk_epa(cudaStream_t stream, gjk_output& output, gjk_input& input)
            {
                if (input.candidate_pair_count == 0) { return true; }

                /*

                mjx/_src/collision_driver.py

                # pair-wise collision functions
                _COLLISION_FUNC = {
                    (GeomType.PLANE, GeomType.SPHERE): plane_sphere,
                    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule,
                    (GeomType.PLANE, GeomType.BOX): plane_convex,
                    (GeomType.PLANE, GeomType.ELLIPSOID): plane_ellipsoid,
                    (GeomType.PLANE, GeomType.CYLINDER): plane_cylinder,
                    (GeomType.PLANE, GeomType.MESH): plane_convex,
                    (GeomType.HFIELD, GeomType.SPHERE): hfield_sphere,
                    (GeomType.HFIELD, GeomType.CAPSULE): hfield_capsule,
                    (GeomType.HFIELD, GeomType.BOX): hfield_convex,
                    (GeomType.HFIELD, GeomType.MESH): hfield_convex,
                    (GeomType.SPHERE, GeomType.SPHERE): sphere_sphere,
                    (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule,
                    (GeomType.SPHERE, GeomType.BOX): sphere_convex,
                    (GeomType.SPHERE, GeomType.MESH): sphere_convex,
                    (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule,
                    (GeomType.CAPSULE, GeomType.BOX): capsule_convex,
                    (GeomType.CAPSULE, GeomType.ELLIPSOID): capsule_ellipsoid,
                    (GeomType.CAPSULE, GeomType.CYLINDER): capsule_cylinder,
                    (GeomType.CAPSULE, GeomType.MESH): capsule_convex,
                    (GeomType.ELLIPSOID, GeomType.ELLIPSOID): ellipsoid_ellipsoid,
                    (GeomType.ELLIPSOID, GeomType.CYLINDER): ellipsoid_cylinder,
                    (GeomType.CYLINDER, GeomType.CYLINDER): cylinder_cylinder,
                    (GeomType.BOX, GeomType.BOX): box_box,
                    (GeomType.BOX, GeomType.MESH): convex_convex,
                    (GeomType.MESH, GeomType.MESH): convex_convex,
                }
                */                

                if(input.key_types.x == mjGEOM_PLANE && input.key_types.y == mjGEOM_SPHERE){ return _gjk_epa<GeomType_PLANE, GeomType_SPHERE>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_PLANE && input.key_types.y == mjGEOM_CAPSULE){ return _gjk_epa<GeomType_PLANE, GeomType_CAPSULE>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_PLANE && input.key_types.y == mjGEOM_BOX){ return _gjk_epa<GeomType_PLANE, GeomType_BOX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_PLANE && input.key_types.y == mjGEOM_ELLIPSOID){ return _gjk_epa<GeomType_PLANE, GeomType_ELLIPSOID>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_PLANE && input.key_types.y == mjGEOM_CYLINDER){ return _gjk_epa<GeomType_PLANE, GeomType_CYLINDER>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_PLANE && input.key_types.y == mjGEOM_MESH){ return _gjk_epa<GeomType_PLANE, GeomType_CONVEX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_SPHERE && input.key_types.y == mjGEOM_SPHERE){ return _gjk_epa<GeomType_SPHERE, GeomType_SPHERE>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_SPHERE && input.key_types.y == mjGEOM_CAPSULE){ return _gjk_epa<GeomType_SPHERE, GeomType_CAPSULE>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_SPHERE && input.key_types.y == mjGEOM_BOX){ return _gjk_epa<GeomType_SPHERE, GeomType_BOX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_SPHERE && input.key_types.y == mjGEOM_MESH){ return _gjk_epa<GeomType_SPHERE, GeomType_CONVEX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_CAPSULE && input.key_types.y == mjGEOM_CAPSULE){ return _gjk_epa<GeomType_CAPSULE, GeomType_CAPSULE>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_CAPSULE && input.key_types.y == mjGEOM_BOX){ return _gjk_epa<GeomType_CAPSULE, GeomType_BOX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_CAPSULE && input.key_types.y == mjGEOM_ELLIPSOID){ return _gjk_epa<GeomType_CAPSULE, GeomType_ELLIPSOID>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_CAPSULE && input.key_types.y == mjGEOM_CYLINDER){ return _gjk_epa<GeomType_CAPSULE, GeomType_CYLINDER>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_CAPSULE && input.key_types.y == mjGEOM_MESH){ return _gjk_epa<GeomType_CAPSULE, GeomType_CONVEX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_ELLIPSOID && input.key_types.y == mjGEOM_ELLIPSOID){ return _gjk_epa<GeomType_ELLIPSOID, GeomType_ELLIPSOID>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_ELLIPSOID && input.key_types.y == mjGEOM_CYLINDER){ return _gjk_epa<GeomType_ELLIPSOID, GeomType_CYLINDER>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_CYLINDER && input.key_types.y == mjGEOM_CYLINDER){ return _gjk_epa<GeomType_CYLINDER, GeomType_CYLINDER>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_BOX && input.key_types.y == mjGEOM_BOX){ return _gjk_epa<GeomType_BOX, GeomType_BOX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_BOX && input.key_types.y == mjGEOM_MESH){ return _gjk_epa<GeomType_BOX, GeomType_CONVEX>(stream, output,  input); }
                if(input.key_types.x == mjGEOM_MESH && input.key_types.y == mjGEOM_MESH){ return _gjk_epa<GeomType_CONVEX, GeomType_CONVEX>(stream, output,  input); }

                return true;
            }
        }
    }
}

#ifdef _ENABLE_XLA_FFI_
#   include <mujoco/mujoco.h>
#   include <driver_types.h>
#   include <xla/ffi/api/ffi.h>


namespace mujoco
{ 
    namespace mjx
    {
        namespace cuda
        {
              xla::ffi::Error LaunchKernel_Calculate_BVH_AABB_dyn(cudaStream_t stream,
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_bvh_aabb_dyn,   
                xla::ffi::Buffer<xla::ffi::DataType::F32> m_aabb,           
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_geom_xpos,      
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_geom_xmat,      
                const unsigned int m_ngeom,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out)
            {
                jax_array _d_bvh_aabb_dyn(jax_array_type_float, d_bvh_aabb_dyn.typed_data(), d_bvh_aabb_dyn.size_bytes(), "");
                jax_array _m_aabb(jax_array_type_float, m_aabb.typed_data(), m_aabb.size_bytes(), "");
                jax_array _d_geom_xpos(jax_array_type_float, d_geom_xpos.typed_data(), d_geom_xpos.size_bytes(), "");
                jax_array _d_geom_xmat(jax_array_type_float, d_geom_xmat.typed_data(), d_geom_xmat.size_bytes(), "");

                // d_bvh_aabb_dyn
                if (!calculate_bvh_aabb_dyn(stream, m_ngeom, _m_aabb, _d_geom_xpos, _d_geom_xmat, _d_bvh_aabb_dyn))
                {
                    xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL, std::string("CUDA error::calculate_bvh_aabb_dyn"));
                }

                cudaDeviceSynchronize();

                return xla::ffi::Error::Success();
            }

            xla::ffi::Error LaunchKernel_CollisionBroadPhase_NxN(cudaStream_t stream,
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_counter_nxn,   
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_pair_nxn,   
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_bvh_aabb_dyn, 
                const unsigned int m_ngeom,
                const unsigned int max_geom_pairs,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out)
            {
                jax_array _col_counter_nxn(jax_array_type_uint, col_counter_nxn.typed_data(), col_counter_nxn.size_bytes(), "");
                jax_array _col_pair_nxn(jax_array_type_uint, col_pair_nxn.typed_data(), col_pair_nxn.size_bytes(), "");
                jax_array _d_bvh_aabb_dyn(jax_array_type_float, d_bvh_aabb_dyn.typed_data(), d_bvh_aabb_dyn.size_bytes(), "");

                // nxn 
                if (!collision_candidate_NxN(stream, m_ngeom, max_geom_pairs, _d_bvh_aabb_dyn, _col_counter_nxn, _col_pair_nxn))
                {
                    xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL, std::string("CUDA error::collision_candidate_NxN"));
                }

                cudaDeviceSynchronize();

                return xla::ffi::Error::Success();
            }

            xla::ffi::Error LaunchKernel_CollisionBroadPhase_Sort(cudaStream_t stream,
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_counter_sort,
                xla::ffi::Buffer<xla::ffi::DataType::U32> col_pair_sort,   
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_bvh_aabb_dyn,  
                xla::ffi::Buffer<xla::ffi::DataType::U32> buffer,          
                const unsigned int m_ngeom,
                const unsigned int max_geom_pairs,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out)
            {   
                jax_array _col_counter_sort(jax_array_type_float, col_counter_sort.typed_data(), col_counter_sort.size_bytes(), "col_counter_sort");
                jax_array _col_pair_sort(jax_array_type_float, col_pair_sort.typed_data(), col_pair_sort.size_bytes(), "col_pair_sort");
                jax_array _d_bvh_aabb_dyn(jax_array_type_float, d_bvh_aabb_dyn.typed_data(), d_bvh_aabb_dyn.size_bytes(), "d_bvh_aabb_dyn");
                jax_array _buffer(jax_array_type_uint, buffer.typed_data(), buffer.size_bytes(), "");
                int axis = 0;

                // [0] get the space min and max coordinates in the axis 
                float spaceMin_axis = 0.0f, spaceMax_axis = 0.0f;
                float spaceAABB_precision = 0.01f;
                float spaceAABB_sizeLimit = 10000.0f;
                if (!(update_spaceAABB(stream, spaceMin_axis, spaceMax_axis, &_buffer, axis, m_ngeom, _d_bvh_aabb_dyn,spaceAABB_sizeLimit, spaceAABB_precision)))
                {
                    xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL, std::string("CUDA error::update_spaceAABB"));
                }

                // nxn 
                if (!collision_candidate_by_sort(stream, m_ngeom,max_geom_pairs, _d_bvh_aabb_dyn, axis, spaceMin_axis, spaceMax_axis,_col_counter_sort, _col_pair_sort, _buffer))
                {
                    xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL, std::string("CUDA error::collision_candidate_by_sort"));
                }

                cudaDeviceSynchronize();

                return xla::ffi::Error::Success();
            }

            xla::ffi::Error LaunchKernel_GJK_EPA(
                cudaStream_t stream,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> d_contact_counter,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_pos,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_dist,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_frame,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_normal,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> d_contact_simplex,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::S32>> d_contact_pairs,
                xla::ffi::Buffer<xla::ffi::DataType::S32> d_candidate_pairs,

                xla::ffi::Buffer<xla::ffi::DataType::F32> d_geom_xpos,
                xla::ffi::Buffer<xla::ffi::DataType::F32> d_geom_xmat,
                xla::ffi::Buffer<xla::ffi::DataType::F32> m_geom_size,
                xla::ffi::Buffer<xla::ffi::DataType::S32> m_geom_dataid,
                xla::ffi::Buffer<xla::ffi::DataType::F32> m_mesh_convex_vert,
                xla::ffi::Buffer<xla::ffi::DataType::S32> m_mesh_convex_vert_offset,
                const unsigned int ncon,
                const unsigned int m_ngeom,
                const unsigned int candidate_pair_count_max,
                const unsigned int candidate_pair_count,
                const int key_types0, 
                const int key_types1,
                const float depthExtension, 
                const unsigned int gjkIterationCount, 
                const unsigned int epaIterationCount, 
                const unsigned int epaBestCount, 
                const unsigned int multiPolygonCount, 
                const float multiTiltAngle,
                const unsigned int compress_result,
                xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::U32>> out)
            {
                jax_array _d_contact_counter(jax_array_type_uint, d_contact_counter->typed_data(), d_contact_counter->size_bytes(), "");
                jax_array _d_contact_pos(jax_array_type_float, d_contact_pos->typed_data(), d_contact_pos->size_bytes(), "");
                jax_array _d_contact_dist(jax_array_type_float, d_contact_dist->typed_data(), d_contact_dist->size_bytes(), "");
                jax_array _d_contact_frame(jax_array_type_float, d_contact_frame->typed_data(), d_contact_frame->size_bytes(), "");

                jax_array _d_contact_normal(jax_array_type_float, d_contact_normal->typed_data(), d_contact_normal->size_bytes(), "");
                jax_array _d_contact_simplex(jax_array_type_float, d_contact_simplex->typed_data(), d_contact_simplex->size_bytes(), "");
                jax_array _d_contact_pairs(jax_array_type_int, d_contact_pairs->typed_data(), d_contact_pairs->size_bytes(), "");

                jax_array _d_geom_xpos(jax_array_type_float, d_geom_xpos.typed_data(), d_geom_xpos.size_bytes(), "");
                jax_array _d_geom_xmat(jax_array_type_float, d_geom_xmat.typed_data(), d_geom_xmat.size_bytes(), "");
                jax_array _d_candidate_pairs(jax_array_type_int, d_candidate_pairs.typed_data(), d_candidate_pairs.size_bytes(), "");
                jax_array _m_geom_size(jax_array_type_float, m_geom_size.typed_data(), m_geom_size.size_bytes(), "");
                jax_array _m_geom_dataid(jax_array_type_int, m_geom_dataid.typed_data(), m_geom_dataid.size_bytes(), "");
                jax_array _m_mesh_convex_vert(jax_array_type_float, m_mesh_convex_vert.typed_data(), m_mesh_convex_vert.size_bytes(), "");
                jax_array _m_mesh_convex_vert_offset(jax_array_type_int, m_mesh_convex_vert_offset.typed_data(), m_mesh_convex_vert_offset.size_bytes(), "");

                gjk_output output; 
                output.d_contact_counter = &_d_contact_counter;
                output.d_contact_pos = &_d_contact_pos;
                output.d_contact_dist = &_d_contact_dist;
                output.d_contact_frame = &_d_contact_frame;
                output.d_contact_normal = &_d_contact_normal;
                output.d_contact_simplex = &_d_contact_simplex;
                output.d_contact_pairs = &_d_contact_pairs;

                gjk_input input;
                input.ncon = ncon;
                input.m_ngeom = m_ngeom;
                input.d_geom_xpos = &_d_geom_xpos;
                input.d_geom_xmat = &_d_geom_xmat;
                input.candidate_pair_count_max = candidate_pair_count_max;
                input.candidate_pair_count = candidate_pair_count;
                input.d_candidate_pairs = &_d_candidate_pairs;
                input.key_types = make_uint2(key_types0, key_types1);
                input.m_geom_dataid = &_m_geom_dataid;
                input.m_mesh_convex_vert = &_m_mesh_convex_vert;
                input.m_mesh_convex_vert_offset = &_m_mesh_convex_vert_offset;
                input.m_geom_size = &_m_geom_size;
                input.depthExtension = depthExtension;
                input.gjkIterationCount = gjkIterationCount;
                input.epaIterationCount = epaIterationCount;
                input.epaBestCount = epaBestCount;
                input.multiPolygonCount = multiPolygonCount;
                input.multiTiltAngle = multiTiltAngle;
                input.compress_result = (compress_result > 0);
                // d_bvh_aabb_dyn
                if (!gjk_epa(stream, output, input))
                {
                    xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL, std::string("CUDA error::gjk_epa"));
                }
                
                cudaDeviceSynchronize();

                return xla::ffi::Error::Success();
            }
        }
    }
}

#endif 
