#include <bulk/bulk.hpp>

#include "tpt/math/stringify.hpp"

#include "communication_structures.hpp"

#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/Globals.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/cuda/3d/mem3d.h>

namespace pleiades {

// TODO implement gather and scatter steps
template <typename T>
void gather(bulk::coarray<T>& red_buf, std::vector<gather_task> tasks, const T* proj_data)
{
    for (auto task : tasks) {
        for (auto [remote, line] : task.lines) {
            auto [begin, count] = line;
            // TODO: is int large enough for these indices? const_cast?
            red_buf(task.owner)[{(int)remote, (int)(remote + count)}] = {
            const_cast<T*>(&proj_data[begin]), count};
        }
    }
    red_buf.world().sync();
}

template <typename T>
void reduce(bulk::coarray<T>& red_buf, std::vector<reduction_task> tasks, T* proj_data)
{
    for (auto [in, count, blocks, out] : tasks) {
        for (auto i = 0u; i < count; ++i) {
            for (auto j = 0u; j < blocks; ++j) {
                proj_data[out + i] += red_buf[in + j * count];
            }
        }
    }
}

template <typename T>
void scatter(bulk::coarray<T>& proj_buf, std::vector<scatter_task> tasks, const T* proj_data)
{
    for (auto task : tasks) {
        for (auto [begins, line] : task.lines) {
            for (auto i = 0u; i < task.contributors.size(); ++i) {
                auto remote = begins[i];
                auto [local, count] = line;
                proj_buf(task.contributors[i])[{(int)remote, (int)(remote + count)}] = {
                const_cast<T*>(&proj_data[local]), count};
            }
        }
        proj_buf.world().sync();
    }
}

astra::SConeProjection get_astra_vectors(const tpt::geometry::base<3_D, float>& g, int index)
{
    tpt::math::vec<3_D, float> src = g.source_location(index);
    tpt::math::vec<3_D, float> det = g.detector_corner(index);
    std::array<tpt::math::vec<3_D, float>, 2> delta = g.projection_delta(index);

    // TODO: Check/fix coordinate order
    // Assumption for now:
    //  volume: x, y, z
    //  detector: v, u

    astra::SConeProjection vec;
    vec.fSrcX = src[0];
    vec.fSrcY = src[1];
    vec.fSrcZ = src[2];
    vec.fDetSX = det[0];
    vec.fDetSY = det[1];
    vec.fDetSZ = det[2];
    vec.fDetUX = delta[1][0];
    vec.fDetUY = delta[1][1];
    vec.fDetUZ = delta[1][2];
    vec.fDetVX = delta[0][0];
    vec.fDetVY = delta[0][1];
    vec.fDetVZ = delta[0][2];

    return vec;
}

astra::SConeProjection get_astra_subvectors(const tpt::geometry::base<3_D, float>& g,
                                            int index,
                                            const geometry_info& gi,
                                            int proc_id)
{
    astra::SConeProjection vec = get_astra_vectors(g, index);

    // translate detector to local corner

    vec.fDetSX += std::get<0>(gi.corner[proc_id][index]) * vec.fDetVX;
    vec.fDetSY += std::get<0>(gi.corner[proc_id][index]) * vec.fDetVY;
    vec.fDetSZ += std::get<0>(gi.corner[proc_id][index]) * vec.fDetVZ;

    vec.fDetSX += std::get<1>(gi.corner[proc_id][index]) * vec.fDetUX;
    vec.fDetSY += std::get<1>(gi.corner[proc_id][index]) * vec.fDetUY;
    vec.fDetSZ += std::get<1>(gi.corner[proc_id][index]) * vec.fDetUZ;

    return vec;
}

astra::CConeVecProjectionGeometry3D*
get_astra_subgeometry(const tpt::geometry::base<3_D, float>& g, const geometry_info& gi, int proc)
{
    astra::SConeProjection* vecs = new astra::SConeProjection[gi.projection_count];
    for (auto i = 0; i < gi.projection_count; ++i)
        vecs[i] = get_astra_subvectors(g, i, gi, proc);
    astra::CConeVecProjectionGeometry3D* geom =
    new astra::CConeVecProjectionGeometry3D(gi.projection_count, std::get<0>(gi.shape),
                                            std::get<1>(gi.shape), vecs);
    delete[] vecs;

    return geom;
}

void reconstruct(bulk::world& world,
                 const tpt::grcb::node<float>& root,
                 tpt::geometry::base<3_D, float>& g,
                 tpt::volume<3_D, float> v)
{
    // ... do some Landweber iterations
    auto s = world.rank();
    auto p = world.active_processors();

    world.log("Computing tasks");

    // prepare tasks
    auto [gathers, scatters, reduces, meta] =
    tasks(world, g, root, tpt::grcb::corners(v));

    world.log("Tasks computed");

    // make projection and reduction buffers
    // what size?
    // should `tasks` return additional metadata?
    // (we probably need geometry_info for generating geometries.)

    auto red_buf = bulk::coarray<float>(world, meta.reduction_size);
    auto proj_buf = bulk::coarray<float>(world, meta.projection_size);

    world.log("Construct geometry info");
    auto g_info = construct_geometry_info(g, p, tpt::grcb::corners(v), root);
    world.log("Geometry info constructed");

    // TODO construct these (local) values
    // make local volume geometry
    auto parts = pleiades::partitioning_to_corners(root, tpt::grcb::corners(v));
    auto [a, b] = tpt::grcb::min_max_cube<float>(parts[s]);
    auto sz = b - a;
    world.log("a %s b %s sz %s size %s voxels %s",
              tpt::math::to_string<3_D, float>(a).c_str(),
              tpt::math::to_string<3_D, float>(b).c_str(),
              tpt::math::to_string<3_D, float>(sz).c_str(),
              tpt::math::to_string<3_D, float>(v.physical_lengths()).c_str(),
              tpt::math::to_string<3_D, int>(v.voxels()).c_str());

    auto nu = (uint32_t)std::get<0>(g_info.local_shape[s]);
    auto nv = (uint32_t)std::get<1>(g_info.local_shape[s]);

    auto nx = (uint32_t)(v.voxels()[0] * (sz[0] / v.physical_lengths()[0]) + 0.5f);
    auto ny = (uint32_t)(v.voxels()[1] * (sz[1] / v.physical_lengths()[1]) + 0.5f);
    auto nz = (uint32_t)(v.voxels()[2] * (sz[2] / v.physical_lengths()[2]) + 0.5f);

    auto np = (uint32_t)g.projection_count();

    world.log("Core stats [%d, %d, %d], {%d, %d, %d}", nx, ny, nz, nu, nv, np);

    world.log("Making ASTRA objects");
    // make ASTRA vol geom
    auto vol_geom =
    astra::CVolumeGeometry3D(nx, ny, nz, a[0], a[1], a[2], b[0], b[1], b[2]);

    // Allocate GPU memory
    auto D_proj = astraCUDA3d::allocateGPUMemory(nu, np, nv, astraCUDA3d::INIT_ZERO);
    auto D_iter = astraCUDA3d::allocateGPUMemory(nx, ny, nz, astraCUDA3d::INIT_ZERO);

    // TODO use this to make buf and to copy from GPU memory
    // astraCUDA3d::SSubDimensions3D dims_vol{nx, ny, nz, nx, nx, ny, nz, 0, 0, 0};
    astraCUDA3d::SSubDimensions3D dims_proj{nu, np, nv, nu, nu,
                                            np, nv, 0,  0,  0};

    astra::CProjectionGeometry3D* proj_geom = get_astra_subgeometry(g, g_info, s);

    world.log("Iterating");
    auto num_iters = 2u;
    for (auto iter = 0u; iter < num_iters; ++iter) {
        world.log("Iteration %i", iter);

        // ASTRA fp (D_iter -> D_proj)
        world.log("zero", iter);
        astraCUDA3d::zeroGPUMemory(D_proj, nu, np, nv);
        world.log("fp", iter);
        astraCUDA3d::FP(proj_geom, D_proj, &vol_geom, D_iter, 1, astraCUDA3d::ker3d_default);

        // download from GPU
        world.log("gpu -> cpu");
        astraCUDA3d::copyFromGPUMemory(proj_buf.begin(), D_proj, dims_proj);

        world.log("gather");
        gather(red_buf, gathers, proj_buf.begin());

        // perform reductions
        world.log("reduce");
        reduce(red_buf, reduces, proj_buf.begin());

        // subtract from b
        // (can do inner products in data space now before scatter (for cgls))
        world.log("scatter");
        scatter(proj_buf, scatters, proj_buf.begin());

        // upload to GPU
        world.log("cpu -> gpu");
        astraCUDA3d::copyToGPUMemory(proj_buf.begin(), D_proj, dims_proj);
        // ASTRA bp (D_proj -> add to D_iter)
        // TODO check last argument
        world.log("bp");
        astraCUDA3d::BP(proj_geom, D_proj, &vol_geom, D_iter, 1, false);
    }

    // TODO make buf
    // Store D_iter
    // astraCUDA3d::copyFromGPUMemory(buf, D_iter, dims_vol);

    astraCUDA3d::freeGPUMemory(D_proj);
    astraCUDA3d::freeGPUMemory(D_iter);
    delete proj_geom;
}

} // namespace pleiades
