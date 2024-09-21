#include <cstring>

namespace mujoco
{
    namespace mjx
    {
        namespace cuda
        {
			
#define GK_CONTACT_DIST_INVALID 100000000.0f
#define GJK_MULTICONTACT_COUNT 4            

			// ------------------------------------------------------------------------------------------------
			// GeomType for GJK 
			// ------------------------------------------------------------------------------------------------
			struct GeomType_PLANE
            {
            };
            struct GeomType_HFIELD
            {
				float width = 0.0f;
				float height = 0.0f;
            };
            struct GeomType_SPHERE
            {
                float radius = 0.0f;
            };
            struct GeomType_CAPSULE
            {
                float radius = 0.0f;
				float halfSize = 0.0f;
            };
            struct GeomType_ELLIPSOID
            {
				float radiusx = 0.0f;
				float radiusy = 0.0f;
				float radiusz = 0.0f;
            };
            struct GeomType_CYLINDER
            {
				float radius = 0.0f;
				float halfSize = 0.0f;
            };
			struct GeomType_BOX
            {
				float halfSizeX = 0.0f;
				float halfSizeY = 0.0f;
				float halfSizeZ = 0.0f;
            };
            struct GeomType_CONVEX
            {
				int offset = 0;
				int count = 0;
            };
			
			// ------------------------------------------------------------------------------------------------
			// GJK_GetGeomSize
			// ------------------------------------------------------------------------------------------------
			template <class GJK_GeomType>  __host__ __device__ __forceinline__  GJK_GeomType GJK_GetGeomSize(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset);

			template <>  __host__ __device__ __forceinline__  GeomType_PLANE GJK_GetGeomSize<GeomType_PLANE>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_PLANE x;
				return x;
			}
			template <>  __host__ __device__ __forceinline__  GeomType_HFIELD GJK_GetGeomSize<GeomType_HFIELD>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_HFIELD x;
				return x;
			}
			template <>  __host__ __device__ __forceinline__  GeomType_SPHERE GJK_GetGeomSize<GeomType_SPHERE>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_SPHERE x;
				x.radius = geom_size[gidx * 3 + 0];
				return x;
			}
			template <>  __host__ __device__ __forceinline__  GeomType_CAPSULE GJK_GetGeomSize<GeomType_CAPSULE>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_CAPSULE x;
				x.radius = geom_size[gidx * 3 + 0];
				x.halfSize = geom_size[gidx * 3 + 1];
				return x;
			}
			template <>  __host__ __device__ __forceinline__  GeomType_ELLIPSOID GJK_GetGeomSize<GeomType_ELLIPSOID>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_ELLIPSOID x;
				x.radiusx = geom_size[gidx * 3 + 0];
				x.radiusy = geom_size[gidx * 3 + 1];
				x.radiusz = geom_size[gidx * 3 + 2];
				return x;
			}
			template <>  __host__ __device__ __forceinline__  GeomType_CYLINDER GJK_GetGeomSize<GeomType_CYLINDER>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_CYLINDER x;
				x.radius = geom_size[gidx * 3 + 0];
				x.halfSize = geom_size[gidx * 3 + 1];
				return x;
			}
			template <>  __host__ __device__ __forceinline__  GeomType_BOX GJK_GetGeomSize<GeomType_BOX>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_BOX x;
				x.halfSizeX = geom_size[gidx * 3 + 0];
				x.halfSizeY = geom_size[gidx * 3 + 1];
				x.halfSizeZ = geom_size[gidx * 3 + 2];
				return x;
			}
			template <>  __host__ __device__ __forceinline__  GeomType_CONVEX GJK_GetGeomSize<GeomType_CONVEX>(const float * __restrict geom_size, const int gidx, const int didx, const int* __restrict d_convex_offset)
			{
				GeomType_CONVEX x;
				if((!d_convex_offset) || (didx < 0)) { x.offset = x.count = 0; return x;} 
				x.offset = d_convex_offset[didx];
				x.count = d_convex_offset[didx + 1] - x.offset;
				return x;
			}

			// ------------------------------------------------------------------------------------------------
			// GJK_SupportPoint
			// ------------------------------------------------------------------------------------------------
			template <class GJK_GeomType> __host__  __device__ __forceinline__ float GJK_SupportPoint(const GJK_GeomType& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex);

			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_PLANE>(const GeomType_PLANE& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex)
			{
				float3 n0 = { matrix[0].x * n.x + matrix[1].x * n.y + matrix[2].x * n.z, matrix[0].y * n.x + matrix[1].y * n.y + matrix[2].y * n.z, matrix[0].z * n.x + matrix[1].z * n.y + matrix[2].z * n.z };

				float n0norm = sqrtf(n0.x * n0.x + n0.y * n0.y);
				float n0x = (n0norm > 0) ? n0.x / n0norm : 1.0f, n0y = (n0norm > 0) ? n0.y / n0norm : 0.0f;

				float largeSize = 5.0f;
				float3 loc = {n0x * largeSize, n0y * largeSize, largeSize * ((n0.z < 0) ? -1.0f : 0.0f) };

				out.x = matrix[0].x * loc.x + matrix[0].y * loc.y + matrix[0].z * loc.z + matrix[0].w;
				out.y = matrix[1].x * loc.x + matrix[1].y * loc.y + matrix[1].z * loc.z + matrix[1].w;
				out.z = matrix[2].x * loc.x + matrix[2].y * loc.y + matrix[2].z * loc.z + matrix[2].w;

                return(out.x * n.x + out.y * n.y + out.z * n.z);
            }
			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_HFIELD>(const GeomType_HFIELD& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex)
			{
				out.x = out.y = out.z = 0.0f;
                return 0.0f;
            }
			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_SPHERE>(const GeomType_SPHERE& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex)
			 {
                out.x = matrix[0].w + prim.radius * n.x;
                out.y = matrix[1].w + prim.radius * n.y;
                out.z = matrix[2].w + prim.radius * n.z;
                return(out.x * n.x + out.y * n.y + out.z * n.z);
            }
			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_CAPSULE>(const GeomType_CAPSULE& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex)
			{
				float3 n0 = { matrix[0].x * n.x + matrix[1].x * n.y + matrix[2].x * n.z, matrix[0].y * n.x + matrix[1].y * n.y + matrix[2].y * n.z, matrix[0].z * n.x + matrix[1].z * n.y + matrix[2].z * n.z };

				float3 loc = prim.radius * n0;
				loc.z += prim.halfSize * ((n0.z >= 0) ? 1.0f : -1.0f);

				out.x = matrix[0].x * loc.x + matrix[0].y * loc.y + matrix[0].z * loc.z + matrix[0].w;
				out.y = matrix[1].x * loc.x + matrix[1].y * loc.y + matrix[1].z * loc.z + matrix[1].w;
				out.z = matrix[2].x * loc.x + matrix[2].y * loc.y + matrix[2].z * loc.z + matrix[2].w;

                return(out.x * n.x + out.y * n.y + out.z * n.z);
            }
			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_ELLIPSOID>(const GeomType_ELLIPSOID& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex)
			{
				float3 n0 = { matrix[0].x * n.x + matrix[1].x * n.y + matrix[2].x * n.z, matrix[0].y * n.x + matrix[1].y * n.y + matrix[2].y * n.z, matrix[0].z * n.x + matrix[1].z * n.y + matrix[2].z * n.z };
				float3 r0 = { prim.radiusx * prim.radiusx * n0.x, prim.radiusy * prim.radiusy * n0.y, prim.radiusz * prim.radiusz * n0.z };
				float3 loc = r0 / sqrtf(dot(r0, n0));
				out.x = matrix[0].x * loc.x + matrix[0].y * loc.y + matrix[0].z * loc.z + matrix[0].w;
				out.y = matrix[1].x * loc.x + matrix[1].y * loc.y + matrix[1].z * loc.z + matrix[1].w;
				out.z = matrix[2].x * loc.x + matrix[2].y * loc.y + matrix[2].z * loc.z + matrix[2].w;
                return(out.x * n.x + out.y * n.y + out.z * n.z);
            }
			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_CYLINDER>(const GeomType_CYLINDER& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex)
			{
				float3 n0 = { matrix[0].x * n.x + matrix[1].x * n.y + matrix[2].x * n.z, matrix[0].y * n.x + matrix[1].y * n.y + matrix[2].y * n.z, matrix[0].z * n.x + matrix[1].z * n.y + matrix[2].z * n.z };
				float n0norm = sqrtf(n0.x * n0.x + n0.y * n0.y);
				float n0x = (n0norm > 0) ? n0.x / n0norm : 1.0f, n0y = (n0norm > 0) ? n0.y / n0norm : 0.0f;
				float3 loc = { prim.radius * n0x, prim.radius * n0y, prim.halfSize * ((n0.z >= 0) ? 1.0f : -1.0f) };
				out.x = matrix[0].x * loc.x + matrix[0].y * loc.y + matrix[0].z * loc.z + matrix[0].w;
				out.y = matrix[1].x * loc.x + matrix[1].y * loc.y + matrix[1].z * loc.z + matrix[1].w;
				out.z = matrix[2].x * loc.x + matrix[2].y * loc.y + matrix[2].z * loc.z + matrix[2].w;

                return(out.x * n.x + out.y * n.y + out.z * n.z);
            }
			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_BOX>(const GeomType_BOX& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vert)
			{
				float3 n0 = { matrix[0].x * n.x + matrix[1].x * n.y + matrix[2].x * n.z, matrix[0].y * n.x + matrix[1].y * n.y + matrix[2].y * n.z, matrix[0].z * n.x + matrix[1].z * n.y + matrix[2].z * n.z };

				float3 loc = { prim.halfSizeX * ((n0.x >= 0) ? 1.0f : -1.0f), prim.halfSizeY * ((n0.y >= 0) ? 1.0f : -1.0f), prim.halfSizeZ * ((n0.z >= 0) ? 1.0f : -1.0f) };
				out.x = matrix[0].x * loc.x + matrix[0].y * loc.y + matrix[0].z * loc.z + matrix[0].w;
				out.y = matrix[1].x * loc.x + matrix[1].y * loc.y + matrix[1].z * loc.z + matrix[1].w;
				out.z = matrix[2].x * loc.x + matrix[2].y * loc.y + matrix[2].z * loc.z + matrix[2].w;

                return(out.x * n.x + out.y * n.y + out.z * n.z);
            }
			template <> __host__  __device__ __forceinline__ float GJK_SupportPoint<GeomType_CONVEX>(const GeomType_CONVEX& prim, const float4* matrix, const float3& n, float3& out, const float * __restrict d_convex_vertex)
			{
				float3 n0 = { matrix[0].x * n.x + matrix[1].x * n.y + matrix[2].x * n.z, matrix[0].y * n.x + matrix[1].y * n.y + matrix[2].y * n.z, matrix[0].z * n.x + matrix[1].z * n.y + matrix[2].z * n.z };
                float max = 0.0f;
                float3 loc = make_float3(0.0f, 0.0f, 0.0f);

                for(int i = 0; i < prim.count; i++)
                {
					int offset = (prim.offset + i) * 3;
                    float3 p = make_float3(d_convex_vertex[offset], d_convex_vertex[offset + 1], d_convex_vertex[offset + 2]);
                    float d = dot(n0, p);
                    if ((!i) || (d > max)) max = d, loc = p;
                }
				out.x = matrix[0].x * loc.x + matrix[0].y * loc.y + matrix[0].z * loc.z + matrix[0].w;
				out.y = matrix[1].x * loc.x + matrix[1].y * loc.y + matrix[1].z * loc.z + matrix[1].w;
				out.z = matrix[2].x * loc.x + matrix[2].y * loc.y + matrix[2].z * loc.z + matrix[2].w;

                return(out.x * n.x + out.y * n.y + out.z * n.z);
            }
			
			template <class GeomType1, class GeomType2> __host__ __device__ __forceinline__ float GJK_SupportPoint(
					const GeomType1& prim1, const float4* matrix1, const GeomType2& prim2, const float4* matrix2, const float3& dir, float3& out, 
				const float * __restrict d_convex_vertex)
			{
				float3 out1, out2;
				float dist1 = GJK_SupportPoint<GeomType1>(prim1, matrix1, dir, out1, d_convex_vertex);
				float3 dir2 = make_float3(-dir.x, -dir.y, -dir.z);
				float dist2 = GJK_SupportPoint<GeomType2>(prim2, matrix2, dir2, out2, d_convex_vertex);

				out = out1 - out2;

				return(dist1 + dist2);
			}

			// ------------------------------------------------------------------------------------------------
			// GJK_Normalize
			// ------------------------------------------------------------------------------------------------
			__host__ __device__ __forceinline__ bool GJK_Normalize(float3& a)
			{
				float norm = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
				if ((norm > 1e-8f) && (norm < 1e12f))
				{
					a /= norm;
					return(true);
				}
				else return(false);
			}

			// ------------------------------------------------------------------------------------------------
			// GJK_MakeFrame
			// ------------------------------------------------------------------------------------------------
			__host__ __device__ __forceinline__ void GJK_MakeFrame(float* frame, float3& a)
			{
				float3 y = make_float3(0.0f, 1.0f, 0.0f), z = make_float3(0.0f, 0.0f, 1.0f);
				float3 b = ((-0.5f < a.y) && (a.y < 0.5f)) ? y : z;
				b = b - a * dot(a, b);

				float mag_b = dot(b, b);
				if(mag_b > 0.0f){ b /= sqrtf(mag_b); }
				if(!(a.x || a.y || a.z)) { b = make_float3(0.0f, 0.0f, 0.0f); }
				float3 c = cross(a, b);

				frame[0] = a.x; frame[3] = b.x; frame[6] = c.x;
				frame[1] = a.y; frame[4] = b.y; frame[7] = c.y;
				frame[2] = a.z; frame[5] = b.z; frame[8] = c.z;
			}

			// ------------------------------------------------------------------------------------------------
			// GJK_EPA_InitContact
			// ------------------------------------------------------------------------------------------------
			// init up to candidate_pair_count_max 
			// initializes the distance of each contact to infinite

			__global__ void GJK_EPA_InitContact(const unsigned int candidate_pair_count_max, const int ncon, float* __restrict d_contact_dist)
			{
				unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
				if (m < candidate_pair_count_max)
				{	
					for(int ci = 0; ci < ncon; ci++)
					{
						d_contact_dist[m * ncon + ci] = GK_CONTACT_DIST_INVALID;
					}
				}
			}

			// ------------------------------------------------------------------------------------------------
			// GJK
			// ------------------------------------------------------------------------------------------------
			// calculates whether two object intersects (if depth > 0)
			// it can be modified to return only the subset which intersects using atomic operations as in EPA (this would work correctly only with no depth extension)
			// gjkIterationCount - the number of iterations (default 10) - increase if it does not converge in default number of steps

			template <class GeomType1, class GeomType2>
			__global__ void GJK(
				const unsigned int candidate_pair_count,
				// output 
				float* __restrict d_contact_normal, float4* __restrict d_simplex, 
				// input 
				const int* __restrict d_candidate_pairs,
				const unsigned int ncon, const unsigned int m_ngeom, const float* __restrict d_geom_size, const float* geom_xpos, const float* geom_xmat, // transformation 
				const int * __restrict m_geom_dataid, const float * __restrict d_convex_vertex, const int * __restrict d_convex_offset,
				const int gjkIterationCount)
			{
				unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
				if (m < candidate_pair_count)
				{
					int pi = d_candidate_pairs[m * 2], pj = d_candidate_pairs[m * 2 + 1];
					if(pi < 0 || pj < 0) {return;}
					
					GeomType1 prim_i = GJK_GetGeomSize<GeomType1>(d_geom_size, pi, m_geom_dataid ? m_geom_dataid[pi] : -1, d_convex_offset);
					GeomType2 prim_j = GJK_GetGeomSize<GeomType2>(d_geom_size, pj, m_geom_dataid ? m_geom_dataid[pj] : -1, d_convex_offset);
			
					float4 mat1[3] = {
						make_float4(geom_xmat[pi * 9 + 0],geom_xmat[pi * 9 + 1],geom_xmat[pi * 9 + 2], geom_xpos[pi * 3 + 0]),
						make_float4(geom_xmat[pi * 9 + 3],geom_xmat[pi * 9 + 4],geom_xmat[pi * 9 + 5], geom_xpos[pi * 3 + 1]),
						make_float4(geom_xmat[pi * 9 + 6],geom_xmat[pi * 9 + 7],geom_xmat[pi * 9 + 8], geom_xpos[pi * 3 + 2])};

					float4 mat2[3] = {
						make_float4(geom_xmat[pj * 9 + 0],geom_xmat[pj * 9 + 1],geom_xmat[pj * 9 + 2], geom_xpos[pj * 3 + 0]),
						make_float4(geom_xmat[pj * 9 + 3],geom_xmat[pj * 9 + 4],geom_xmat[pj * 9 + 5], geom_xpos[pj * 3 + 1]),
						make_float4(geom_xmat[pj * 9 + 6],geom_xmat[pj * 9 + 7],geom_xmat[pj * 9 + 8], geom_xpos[pj * 3 + 2])};

					float3 dir = make_float3(0.0f, 0.0f, 1.0f), normal, simplex[4];
					float3 dir_n = -dir;
					float depth = 1e30f;

					float max = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, dir, simplex[0], d_convex_vertex);
					float min = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, dir_n, simplex[1], d_convex_vertex);

					if (max < min) depth = max, normal = dir;
					else depth = min, normal = dir_n;

					float3 diff = simplex[0] - simplex[1];
					GJK_Normalize(diff);

					if ((fabs(diff.x) < fabs(diff.y)) && (fabs(diff.x) < fabs(diff.z))) dir = make_float3(1.0f - diff.x * diff.x, -diff.x * diff.y, -diff.x * diff.z);
					else if (fabs(diff.y) < fabs(diff.z)) dir = make_float3(-diff.y * diff.x, 1.0f - diff.y * diff.y, -diff.y * diff.z);
					else dir = make_float3(-diff.z * diff.x, -diff.z * diff.y, 1.0f - diff.z * diff.z);
					GJK_Normalize(dir);

					max = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, dir, simplex[3], d_convex_vertex);
					min = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, dir_n, simplex[2], d_convex_vertex);

					if (max < depth) depth = max, normal = dir;
					if (min < depth) depth = min, normal = dir_n;

					for (int i = 0; i < gjkIterationCount; i++)
					{
						float3 plane[4];
						float d[4];

						plane[0] = cross(simplex[3] - simplex[2], simplex[1] - simplex[2]);
						plane[1] = cross(simplex[3] - simplex[0], simplex[2] - simplex[0]);
						plane[2] = cross(simplex[3] - simplex[1], simplex[0] - simplex[1]);
						plane[3] = cross(simplex[2] - simplex[0], simplex[1] - simplex[0]);

						d[0] = (GJK_Normalize(plane[0])) ? dot(plane[0], simplex[2]) : 1e30f;
						d[1] = (GJK_Normalize(plane[1])) ? dot(plane[1], simplex[0]) : 1e30f;
						d[2] = (GJK_Normalize(plane[2])) ? dot(plane[2], simplex[1]) : 1e30f;
						d[3] = (GJK_Normalize(plane[3])) ? dot(plane[3], simplex[0]) : 1e30f;

						int i1 = (d[0] < d[1]) ? 0 : 1, i2 = (d[2] < d[3]) ? 2 : 3;
						int index = (d[i1] < d[i2]) ? i1 : i2;
						if (d[index] > 0.0f) break;

						float dist = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, plane[index], simplex[index], d_convex_vertex);
						if (dist < depth) depth = dist, normal = plane[index];

						int index1 = (index + 1) & 3, index2 = (index + 2) & 3;
    					float3 swap = simplex[index1];
        				simplex[index1] = simplex[index2], simplex[index2] = swap;
        				if (dist < 0) break;
					}
					float4 *simplex_f4 = (float4 *)simplex;
					d_simplex[m * 3 + 0] = simplex_f4[0];
					d_simplex[m * 3 + 1] = simplex_f4[1];
					d_simplex[m * 3 + 2] = simplex_f4[2];

					if(d_contact_normal) d_contact_normal[m * 3] = normal.x, d_contact_normal[m * 3 + 1] = normal.y, d_contact_normal[m * 3 + 2] = normal.z;
				}
			}

			// ------------------------------------------------------------------------------------------------
			// EPA
			// ------------------------------------------------------------------------------------------------
			// returns the normal and distance (-depth) for each pair - the smallest depth if they intersect, the closest points if objects don't
			// the method returns only the subset of points with (depth > -depthExtension), currently depthExtension is set to inf
			// epaIterationCount - the number of iterations (increase to get a more precise result)
			// epaBestCount - the number of best candidates in each iteration (increase to get a more precise result)
			//				- epaBestCount must be less or equal to const int maxEpaBestCount (which should be as small as possible to minimize the number of required registers)
			//              - if you need a large possible range, compile multiple versions with additional template parameter for maxEpaBestCount
			// compress_result - returns only candidates with (depth > -depthExtension), otherwise it sets the flags activeContacts 0/1
			// exactNegDistance - if flag is on, the algorithm calculates the distance of non-intersecting objects exactly
			//                    for intersecting, the closest point on the Minkowski maniforld to the	origin is always on the face
			//                    for non-intersecting, the closes point can be on the edge or in the corner and in that case additional tests have to be done							

			template <class GeomType1, class GeomType2>
			__global__ void EPA(
				const unsigned int candidate_pair_count,
				// output 
				unsigned int* __restrict activeContacts,
				float* __restrict d_contact_dist, float* __restrict d_contact_frame, const float* __restrict d_contact_normal, int* __restrict d_contact_pairs,
				const float4* __restrict d_simplex, 
				// input 
				const int* __restrict d_candidate_pairs,
				const unsigned int ncon, const unsigned int m_ngeom, const float* __restrict d_geom_size, const float* geom_xpos, const float* geom_xmat, // transformation 
				const int * __restrict m_geom_dataid, const float * __restrict d_convex_vertex, const int * __restrict d_convex_offset,
				const float depthExtension, const int epaIterationCount, const int epaBestCount,
				const bool compress_result)
			{
				const int maxEpaBestCount = 8;
				bool exactNegDistance = true;

				unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
				if (m < candidate_pair_count)
				{
					int pi = d_candidate_pairs[m * 2], pj = d_candidate_pairs[m * 2 + 1];
					if(pi < 0 || pj < 0) {return;}
					
					GeomType1 prim_i = GJK_GetGeomSize<GeomType1>(d_geom_size, pi, m_geom_dataid ? m_geom_dataid[pi] : -1, d_convex_offset);
					GeomType2 prim_j = GJK_GetGeomSize<GeomType2>(d_geom_size, pj, m_geom_dataid ? m_geom_dataid[pj] : -1, d_convex_offset);
			
					float4 mat1[3] = {
						make_float4(geom_xmat[pi * 9 + 0],geom_xmat[pi * 9 + 1],geom_xmat[pi * 9 + 2], geom_xpos[pi * 3 + 0]),
						make_float4(geom_xmat[pi * 9 + 3],geom_xmat[pi * 9 + 4],geom_xmat[pi * 9 + 5], geom_xpos[pi * 3 + 1]),
						make_float4(geom_xmat[pi * 9 + 6],geom_xmat[pi * 9 + 7],geom_xmat[pi * 9 + 8], geom_xpos[pi * 3 + 2])};

					float4 mat2[3] = {
						make_float4(geom_xmat[pj * 9 + 0],geom_xmat[pj * 9 + 1],geom_xmat[pj * 9 + 2], geom_xpos[pj * 3 + 0]),
						make_float4(geom_xmat[pj * 9 + 3],geom_xmat[pj * 9 + 4],geom_xmat[pj * 9 + 5], geom_xpos[pj * 3 + 1]),
						make_float4(geom_xmat[pj * 9 + 6],geom_xmat[pj * 9 + 7],geom_xmat[pj * 9 + 8], geom_xpos[pj * 3 + 2])};

					float3 simplex[4];
					float3 normal = make_float3(d_contact_normal[m * 3], d_contact_normal[m * 3 + 1], d_contact_normal[m * 3 + 2]);
					float depth = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, normal, simplex[0], d_convex_vertex);
					if (depth < -depthExtension) {return; }

					float4* simplex_f4 = (float4 *)simplex;
					simplex_f4[0] = d_simplex[m * 3 + 0];
					simplex_f4[1] = d_simplex[m * 3 + 1];
					simplex_f4[2] = d_simplex[m * 3 + 2];

					if (exactNegDistance)
					{
						for (int i = 0; i < 6; i++)
						{
							int i1 = (i < 3) ? 0 : ((i < 5) ? 1 : 2), i2 = (i < 3) ? i + 1 : (i < 5) ? i - 1 : 4;
							if ((simplex[i1].x != simplex[i2].x) || (simplex[i1].y != simplex[i2].y) || (simplex[i1].z != simplex[i2].z))
							{
								float3 v = simplex[i1] - simplex[i2];
								float alpha = dot(simplex[i1], v) / (v.x * v.x + v.y * v.y + v.z * v.z);
								float3 p0 = ((alpha < 0.0f) ? 0.0f : ((alpha > 1.0f) ? 1.0f : alpha)) * v - simplex[i1];
								if (GJK_Normalize(p0))
								{
									float dist2 = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, p0, v, d_convex_vertex);
									if (dist2 < depth) depth = dist2, normal = p0;
								}
							}
						}
					}

					float3 tr[3 * maxEpaBestCount], tr2[3 * maxEpaBestCount], p[maxEpaBestCount], * triangles = tr, * nextTriangles = tr2;
					float dists[maxEpaBestCount * 3];

					triangles[0] = simplex[2], triangles[1] = simplex[1], triangles[2] = simplex[3];
					triangles[3] = simplex[0], triangles[4] = simplex[2], triangles[5] = simplex[3];
					triangles[6] = simplex[1], triangles[7] = simplex[0], triangles[8] = simplex[3];
					triangles[9] = simplex[0], triangles[10] = simplex[1], triangles[11] = simplex[2];

					int count = 4;
					for (int q = 0; q < epaIterationCount; q++)
					{
						for (int i = 0; i < count; i++)
						{
							float3* triangle = triangles + 3 * i;
							float3 n = cross(triangle[2] - triangle[0], triangle[1] - triangle[0]);

							if (GJK_Normalize(n))
							{
								float dist = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, n, p[i], d_convex_vertex);
								if (dist < depth) depth = dist, normal = n;

								for (int j = 0; j < 3; j++)
								{
									if (exactNegDistance)
									{
										if ((p[i].x != triangle[j].x) || (p[i].y != triangle[j].y) || (p[i].z != triangle[j].z))
										{
											float3 v = p[i] - triangle[j];
											float alpha = dot(p[i], v) / (v.x * v.x + v.y * v.y + v.z * v.z);
											float3 p0 = ((alpha < 0.0f) ? 0.0f : ((alpha > 1.0f) ? 1.0f : alpha)) * v - p[i];
											if (GJK_Normalize(p0))
											{
												float dist2 = GJK_SupportPoint<GeomType1, GeomType2>(prim_i, mat1, prim_j, mat2, p0, v, d_convex_vertex);
												if (dist2 < depth) depth = dist2, normal = p0;
											}
										}
									}
									float3 plane = cross(p[i] - triangle[j], triangle[((j + 1) % 3)] - triangle[j]);
									float d = (GJK_Normalize(plane)) ? dot(plane, triangle[j]) : 1e30f;

									dists[i * 3 + j] = (((d < 0) && (depth >= 0)) || ((triangle[((j + 2) % 3)].x == p[i].x) && (triangle[((j + 2) % 3)].y == p[i].y) && (triangle[((j + 2) % 3)].z == p[i].z))) ? 1e30f : d;
								}
							}
							else for (int j = 0; j < 3; j++) dists[i * 3 + j] = 2e30f;
						}
						int prevCount = count;
						count = (count * 3 < epaBestCount) ? count * 3 : epaBestCount;

						for (int j = 0; j < count; j++)
						{
							int bestIndex = 0;
							float d = dists[0];
							for (int i = 1; i < 3 * prevCount; i++) if (dists[i] < d) d = dists[i], bestIndex = i;
							dists[bestIndex] = 2e30f;

							int parentIndex = bestIndex / 3, childIndex = bestIndex % 3;
							float3* triangle = nextTriangles + j * 3;
							triangle[0] = triangles[parentIndex * 3 + childIndex], triangle[1] = triangles[parentIndex * 3 + ((childIndex + 1) % 3)], triangle[2] = p[parentIndex];
						}
						float3* swap = triangles;
						triangles = nextTriangles, nextTriangles = swap;
					}
					if (depth < -depthExtension) {return; }

 					int mIdx = m;	// mIdx: index in the memory 
					if(compress_result) { mIdx = atomicAdd(&activeContacts[0], 1); }
					else { activeContacts[m] = 1; }

					for (int nci = 0; nci < ncon; nci++)
					{	
						const int offset = (mIdx * ncon + nci);
						d_contact_dist[offset] = -depth;
					}

					if(d_contact_frame)
					{
						float frame[9]; GJK_MakeFrame(frame, normal);
						for (int nci = 0; nci < ncon; nci++)
						{	
							const int offset = (mIdx * ncon + nci) * 9;
							for(int k = 0; k < 9; k++){ d_contact_frame[offset + k] = frame[k];}
						}
					}
				
					if(d_contact_pairs)
					{
						d_contact_pairs[mIdx * 2] = pi;
						d_contact_pairs[mIdx * 2 + 1] = pj;
					}
				}
			}


			// ------------------------------------------------------------------------------------------------
			// GetMultipleContacts
			// ------------------------------------------------------------------------------------------------
			// calculates multiple contact points given the normal from EPA
			//		1. calculates the polygon on each shape by tilting the normal "multiTiltAngle" degrees in the orthogonal complement of the normal
			//			- the multiTiltAngle can be changed to depend on the depth of the contact (for example linear dependence)
			//		2. the normal is tilted "multiPolygonCount" in the directions around (works well for >= 6, default is 8)
			//			- multiPolygonCount must be less of equal maxMultiPolygonCount (which should be as small as possible to minimize the number of required registers)
			//		3. the intersection between these two polygons in calculated in the 2D space (complement to the normal)
			//			- if they intersect (they should always if there are no numerical issues) - extreme points in both directions are found 
			//				- this can be modified to the extremes in the direction of eigenvectors of the variance of points of each polygon
			//			- if they don't intersect, the closes points of both polygons are found

			template <class GeomType1, class GeomType2>
			__global__ void GetMultipleContacts(
				const unsigned int contact_pair_count,
				// output 
				float* __restrict d_contact_pos,
				// input 
				 const float* __restrict d_contact_dist, const float* __restrict d_contact_frame,
				const int* __restrict contact_pair_list,
				const unsigned int ncon, const unsigned int m_ngeom, const float* __restrict d_geom_size, const float* geom_xpos, const float* geom_xmat, // transformation 
				const int * __restrict m_geom_dataid, const float * __restrict d_convex_vertex, const int * __restrict d_convex_offset,
				const float depthExtension, const int multiPolygonCount, const float multiTiltAngle)
			{
				const int maxMultiPolygonCount = 8;

				unsigned int mIdx = blockIdx.x * blockDim.x + threadIdx.x;
				if (mIdx < contact_pair_count)
				{
					float depth = -d_contact_dist[mIdx * ncon];
					if(depth < -depthExtension) return;

          float3 normal = make_float3(d_contact_frame[mIdx * ncon * 9], d_contact_frame[mIdx * ncon * 9 + 1], d_contact_frame[mIdx * ncon * 9 + 2]);

					int pi = contact_pair_list[mIdx * 2], pj = contact_pair_list[mIdx * 2 + 1];
					if(pi < 0 || pj < 0) {return;}
					
					GeomType1 prim_i = GJK_GetGeomSize<GeomType1>(d_geom_size, pi, m_geom_dataid ? m_geom_dataid[pi] : -1, d_convex_offset);
					GeomType2 prim_j = GJK_GetGeomSize<GeomType2>(d_geom_size, pj, m_geom_dataid ? m_geom_dataid[pj] : -1, d_convex_offset);
			
					float4 mat1[3] = {
						make_float4(geom_xmat[pi * 9 + 0],geom_xmat[pi * 9 + 1],geom_xmat[pi * 9 + 2], geom_xpos[pi * 3 + 0]),
						make_float4(geom_xmat[pi * 9 + 3],geom_xmat[pi * 9 + 4],geom_xmat[pi * 9 + 5], geom_xpos[pi * 3 + 1]),
						make_float4(geom_xmat[pi * 9 + 6],geom_xmat[pi * 9 + 7],geom_xmat[pi * 9 + 8], geom_xpos[pi * 3 + 2])};

					float4 mat2[3] = {
						make_float4(geom_xmat[pj * 9 + 0],geom_xmat[pj * 9 + 1],geom_xmat[pj * 9 + 2], geom_xpos[pj * 3 + 0]),
						make_float4(geom_xmat[pj * 9 + 3],geom_xmat[pj * 9 + 4],geom_xmat[pj * 9 + 5], geom_xpos[pj * 3 + 1]),
						make_float4(geom_xmat[pj * 9 + 6],geom_xmat[pj * 9 + 7],geom_xmat[pj * 9 + 8], geom_xpos[pj * 3 + 2])};

					float3 dir;
					if ((fabs(normal.x) < fabs(normal.y)) && (fabs(normal.x) < fabs(normal.z))) dir = make_float3(1.0f - normal.x * normal.x, -normal.x * normal.y, -normal.x * normal.z);
					else if (fabs(normal.y) < fabs(normal.z)) dir = make_float3(-normal.y * normal.x, 1.0f - normal.y * normal.y, -normal.y * normal.z);
					else dir = make_float3(-normal.z * normal.x, -normal.z * normal.y, 1.0f - normal.z * normal.z);
					GJK_Normalize(dir);

					float3 dir2 = cross(normal, dir);
					float angle = multiTiltAngle * 3.14159265358979f / 180.0f;
					float c = (float) cos(angle), s = (float) sin(angle), t = 1 - c;

					float3 v1[maxMultiPolygonCount], v2[maxMultiPolygonCount];

					int v1count = 0, v2count = 0;
					for (int i = 0; i < multiPolygonCount; i++)
					{
						float3 axis = cos(2 * i * 3.141592653589f / multiPolygonCount) * dir + sin(2 * i * 3.141592653589f / multiPolygonCount) * dir2;

						float mat[12];
						mat[0] = c + axis.x * axis.x * t, mat[5] = c + axis.y * axis.y * t, mat[10] = c + axis.z * axis.z * t;
						float t1 = axis.x * axis.y * t, t2 = axis.z * s;
						mat[4] = t1 + t2, mat[1] = t1 - t2;
						t1 = axis.x * axis.z * t, t2 = axis.y * s;
						mat[8] = t1 - t2, mat[2] = t1 + t2;
						t1 = axis.y * axis.z * t, t2 = axis.x * s;
						mat[9] = t1 + t2, mat[6] = t1 - t2;

						float3 n, p;

						n.x = mat[0] * normal.x + mat[1] * normal.y + mat[2] * normal.z;
						n.y = mat[4] * normal.x + mat[5] * normal.y + mat[6] * normal.z;
						n.z = mat[8] * normal.x + mat[9] * normal.y + mat[10] * normal.z;

						GJK_SupportPoint<GeomType1>(prim_i, mat1, n, p, d_convex_vertex);
						v1[v1count] = make_float3(dot(p, dir), dot(p, dir2), dot(p, normal));
						if ((!i) || (v1[v1count].x != v1[v1count - 1].x) || (v1[v1count].y != v1[v1count - 1].y) || (v1[v1count].z != v1[v1count - 1].z)) v1count++;

						n = -n;
						GJK_SupportPoint<GeomType2>(prim_j, mat2, n, p, d_convex_vertex);
						v2[v2count] = make_float3(dot(p, dir), dot(p, dir2), dot(p, normal));
						if ((!i) || (v2[v2count].x != v2[v2count - 1].x) || (v2[v2count].y != v2[v2count - 1].y) || (v2[v2count].z != v2[v2count - 1].z)) v2count++;
					}

					if ((v1count > 1) && (v1[v1count - 1].x == v1[0].x) && (v1[v1count - 1].y == v1[0].y) && (v1[v1count - 1].z == v1[0].z)) v1count--;
					if ((v2count > 1) && (v2[v2count - 1].x == v2[0].x) && (v2[v2count - 1].y == v2[0].y) && (v2[v2count - 1].z == v2[0].z)) v2count--;

					float3 out[GJK_MULTICONTACT_COUNT];
					int candCount = 0;

					if (v2count > 1) for (int i = 0; i < v1count; i++)
					{
						float3 m1a = v1[i];
						bool in = true;

						for (int j = 0; j < v2count; j++)
						{
							int i2 = (j + 1) % v2count;
							in &= ((v2[i2].x - v2[j].x) * (m1a.y - v2[j].y) - (v2[i2].y - v2[j].y) * (m1a.x - v2[j].x) >= 0.0f);
							if (!in) break;
						}
						if (in)
						{
							if ((!candCount) || (m1a.x < out[0].x)) out[0] = m1a;
							if ((!candCount) || (m1a.x > out[1].x)) out[1] = m1a;
							if ((!candCount) || (m1a.y < out[2].y)) out[2] = m1a;
							if ((!candCount) || (m1a.y > out[3].y)) out[3] = m1a;
							candCount++;
						}
					}
					if (v1count > 1) for (int i = 0; i < v2count; i++)
					{
						float3 m1a = v2[i];
						bool in = true;

						for (int j = 0; j < v1count; j++)
						{
							int i2 = (j + 1) % v1count;
							in &= ((v1[i2].x - v1[j].x) * (m1a.y - v1[j].y) - (v1[i2].y - v1[j].y) * (m1a.x - v1[j].x) >= 0.0f);
							if (!in) break;
						}
						if (in)
						{
							if ((!candCount) || (m1a.x < out[0].x)) out[0] = m1a;
							if ((!candCount) || (m1a.x > out[1].x)) out[1] = m1a;
							if ((!candCount) || (m1a.y < out[2].y)) out[2] = m1a;
							if ((!candCount) || (m1a.y > out[3].y)) out[3] = m1a;
							candCount++;
						}
					}

					if ((v1count > 1) && (v2count > 1)) for (int i = 0; i < v1count; i++) for (int j = 0; j < v2count; j++)
					{
						float3 m1a = v1[i], m1b = v1[(i + 1) % v1count];
						float3 m2a = v2[j], m2b = v2[(j + 1) % v2count];

						float det = (m2a.y - m2b.y) * (m1b.x - m1a.x) - (m1a.y - m1b.y) * (m2b.x - m2a.x);
						if (fabs(det) > 1e-12f)
						{
							float a12 = (m2b.x - m2a.x) / det, a22 = (m1b.x - m1a.x) / det;
							float a21 = (m1a.y - m1b.y) / det, a11 = (m2a.y - m2b.y) / det;
							float b1 = m2a.x - m1a.x, b2 = m2a.y - m1a.y;

							float alpha = a11 * b1 + a12 * b2, beta = a21 * b1 + a22 * b2;
							if ((alpha >= 0.0f) && (alpha <= 1.0f) && (beta >= 0.0f) && (beta <= 1.0f))
							{
								float3 m0 = make_float3(m1a.x + alpha * (m1b.x - m1a.x), m1a.y + alpha * (m1b.y - m1a.y), (m1a.z + alpha * (m1b.z - m1a.z) + m2a.z + beta * (m2b.z - m2a.z)) * 0.5f);
								if ((!candCount) || (m0.x < out[0].x)) out[0] = m0;
								if ((!candCount) || (m0.x > out[1].x)) out[1] = m0;
								if ((!candCount) || (m0.y < out[2].y)) out[2] = m0;
								if ((!candCount) || (m0.y > out[3].y)) out[3] = m0;
								candCount++;
							}
						}
					}

					float3 multiContacts[GJK_MULTICONTACT_COUNT];
					float3 var_rx;

					if (candCount > 0)
					{
						for (int k = 0; k < GJK_MULTICONTACT_COUNT; k++) multiContacts[k] = out[k].x * dir + out[k].y * dir2 + out[k].z * normal;
						var_rx = (multiContacts[0] + multiContacts[1] + multiContacts[2] + multiContacts[3]) * 0.25f;
					}
					else
					{
						float minDist = 0.0f;
						for (int i = 0; i < v1count; i++) for (int j = 0; j < v2count; j++)
						{
							float3 m1 = v1[i], m2 = v2[j];
							float d = (m1.x - m2.x) * (m1.x - m2.x) + (m1.y - m2.y) * (m1.y - m2.y);
							if (((!i) && (!j)) || (d < minDist))
							{
								minDist = d;
								var_rx = ((m1.x + m2.x) * dir + (m1.y + m2.y) * dir2 + (m1.z + m2.z) * normal) * 0.5f;
							}

							float3 m1b = v1[(i + 1) % v1count], m2b = v2[(j + 1) % v2count];
							if (v1count > 1)
							{
								float d = (m1b.x - m1.x) * (m1b.x - m1.x) + (m1b.y - m1.y) * (m1b.y - m1.y);
								float t = ((m2.y - m1.y) * (m1b.x - m1.x) - (m2.x - m1.x) * (m1b.y - m1.y)) / d;
								float dx = m2.x + (m1b.y - m1.y) * t, dy = m2.y - (m1b.x - m1.x) * t;
								float dist = (dx - m2.x) * (dx - m2.x) + (dy - m2.y) * (dy - m2.y);

								if ((dist < minDist) && ((dx - m1.x) * (m1b.x - m1.x) + (dy - m1.y) * (m1b.y - m1.y) >= 0) && ((dx - m1b.x) * (m1.x - m1b.x) + (dy - m1b.y) * (m1.y - m1b.y) >= 0))
								{
									float alpha = (float) sqrt(((dx - m1.x) * (dx - m1.x) + (dy - m1.y) * (dy - m1.y)) / d);
									minDist = dist;
									float3 w = ((1 - alpha) * m1 + alpha * m1b + m2) * 0.5f;
									var_rx = w.x * dir + w.y * dir2 + w.z * normal;
								}
							}
							if (v2count > 1)
							{
								float d = (m2b.x - m2.x) * (m2b.x - m2.x) + (m2b.y - m2.y) * (m2b.y - m2.y);
								float t = ((m1.y - m2.y) * (m2b.x - m2.x) - (m1.x - m2.x) * (m2b.y - m2.y)) / d;
								float dx = m1.x + (m2b.y - m2.y) * t, dy = m1.y - (m2b.x - m2.x) * t;
								float dist = (dx - m1.x) * (dx - m1.x) + (dy - m1.y) * (dy - m1.y);

								if ((dist < minDist) && ((dx - m2.x) * (m2b.x - m2.x) + (dy - m2.y) * (m2b.y - m2.y) >= 0) && ((dx - m2b.x) * (m2.x - m2b.x) + (dy - m2b.y) * (m2.y - m2b.y) >= 0))
								{
									float alpha = (float) sqrt(((dx - m2.x) * (dx - m2.x) + (dy - m2.y) * (dy - m2.y)) / d);
									minDist = dist;
									float3 w = (m1 + (1 - alpha) * m2 + alpha * m2b) * 0.5f;
									var_rx = w.x * dir + w.y * dir2 + w.z * normal;
								}
							}
						}
						for (int k = 0; k < GJK_MULTICONTACT_COUNT; k++) multiContacts[k] = var_rx;
					}

					if(ncon > 0)
					{
						for(int nci = 0; nci < ncon; nci++)
						{
							const int offset = (mIdx * ncon + nci) * 3;

							d_contact_pos[offset + 0] = (ncon == 1) ? var_rx.x : multiContacts[nci].x;
							d_contact_pos[offset + 1] = (ncon == 1) ? var_rx.y : multiContacts[nci].y;
							d_contact_pos[offset + 2] = (ncon == 1) ? var_rx.z : multiContacts[nci].z;
						}
					}
				}
			}
        }
    }
}
