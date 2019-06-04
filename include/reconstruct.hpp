#include <bulk/bulk.hpp>

#include "communication_structures.hpp"

//#include <astra/cuda/3d/mem3d.h>

namespace pleiades {

// TODO implement gather and scatter steps
template <typename T>
void gather(bulk::coarray<T> red_buf, std::vector<gather_task> tasks, const T* proj_data)
{
    for (auto task : tasks) {
        for (auto [remote, line] : task.lines) {
            auto [begin, count] = line;
            red_buf(task.owner)[{remote, remote + count}] = {&proj_data[begin], count};
        }
    }
    buffer.world.sync();
}

template <typename T>
void reduce(bulk::coarray<T> red_buf, std::vector<scatter_task> tasks, T* proj_data)
{
    // reduce from red_buf into proj_data
    // does scatter give sufficient info?
}

template <typename T>
void scatter(bulk::coarray<T> proj_buf, std::vector<scatter_task> tasks, const T* proj_data)
{
    for (auto task : tasks) {
        for (auto [begins, line] : task.lines) {
            for (auto i = 0u; i < task.contributors.size(); ++i) {
                auto remote = begins[i];
                auto [local, count] = line;
                buffer(task.contributors[i])[{remote, remote + count}] = {&data[local], count};
            }
        }
        buffer.world.sync();
    }
}

void reconstruct(bulk::world& world,
                 const tpt::grcb::node<float>& root,
                 tpt::geometry::base<3_D, float>& g,
                 tpt::volume<3_D, float> v)
{
    // ... do some Landweber iterations for simplicity

    // prepare tasks
    auto [gathers, scatters, meta] = tasks(world, g, root, tpt::grcb::corners(v));

    // make projection and reduction buffers
    // what size?
    // should `tasks` return additional metadata?

    auto red_buf = bulk::coarray<float>(world, meta.reduction_size);
    auto proj_buf = bulk::coarray<float>(world, meta.projection_size);

    // Allocate GPU memory
    // auto D_proj = astraCUDA3d::allocateGPUMemory(nu, g.get_projection_count(), nv, astraCUDA3d::INIT_ZERO);
    // auto D_iter = astraCUDA3d::allocateGPUMemory(nx, ny, nz, astraCUDA3d::INIT_ZERO);
    // astraCUDA3d::SSubDimensions3D dims_vol{nx, ny, nz, nx, nx, ny, nz, 0, 0, 0};
    // astraCUDA3d::SSubDimensions3D dims_proj{nu, g.get_projection_count(), nv, nu, nu, g.get_projection_count(), nv, 0, 0, 0};

    auto num_iters = 100u;
    for (auto iter = 0u; iter < num_iters; ++iter) {

        // ASTRA fp (D_iter -> D_proj)
        // astraCUDA3d::zeroGPUMemory(D_proj, nu, g.get_projection_count(), nv);
        // astraCUDA3d::FP(proj_geom, D_proj, vol_geom, D_iter, 1, astraCUDA3d::ker3d_default);

        // download from GPU
        // astraCUDA3d::copyFromGPUMemory(proj_buf.data(), D_proj, dims_proj);

        gather(red_buf, gathers, proj_buf.data());
        // TODO perform reductions
        reduce(red_buf, proj_buf.data(), scatters);

        // subtract from b
        // (can do inner products in data space now before scatter (for cgls))
        scatter(proj_buf, gathers, proj_buf.data());

        // upload to GPU
        // astraCUDA3d::copyToGPUMemory(proj_buf.data(), D_proj, dims_proj);
        // ASTRA bp (D_proj -> add to D_iter)
        // astraCUDA3d::BP(proj_geom, D_proj, vol_geom, D_iter, 1);
    }

    // Store D_iter
    // astraCUDA3d::copyFromGPUMemory(buf, D_iter, dims_vol);

    // astraCUDA3d::freeGPUMemory(D_proj);
    // astraCUDA3d::freeGPUMemory(D_iter);
}

} // namespace pleiades
