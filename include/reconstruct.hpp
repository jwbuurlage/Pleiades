#include <fstream>

#include <bulk/bulk.hpp>

#include "tpt/math/stringify.hpp"

#include "communication_structures.hpp"
#include "util.hpp"

#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/Globals.h>
#include <astra/Logging.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/cuda/3d/mem3d.h>

namespace pleiades {

template <typename T>
void gather(bulk::coarray<T>& red_buf, const std::vector<gather_task>& tasks, T* proj_data)
{
    for (auto task : tasks) {
        for (auto [remote, line] : task.lines) {
            auto [begin, count] = line;
            red_buf(task.owner)[{remote, remote + count}] = {&proj_data[begin], count};
        }
    }
    red_buf.world().sync();
}

template <typename T>
void reduce(bulk::coarray<T>& red_buf, const std::vector<reduction_task>& tasks, T* proj_data)
{
    auto flops = 0u;
    for (auto [in, count, blocks, out] : tasks) {
        flops += count * blocks;
        for (auto i = 0u; i < count; ++i) {
            for (auto j = 0u; j < blocks; ++j) {
                proj_data[out + i] += red_buf[in + i + j * count];
            }
        }
    }
}

template <typename T>
void scatter(bulk::coarray<T>& proj_buf, const std::vector<scatter_task>& tasks, T* proj_data)
{
    for (auto task : tasks) {
        for (auto [begins, line] : task.lines) {
            for (auto i = 0u; i < task.contributors.size(); ++i) {
                auto remote = begins[i];
                auto [local, count] = line;
                proj_buf(task.contributors[i])[{remote, remote + count}] = {&proj_data[local],
                                                                            count};
            }
        }
    }
    proj_buf.world().sync();
}

/**
 * Run a reconstruction using the communicator `world` with a partitioning given
 * by `root` for the acquisition geometry `g` and object volume `v`.
 *
 * We forward project a cube phantom and use this simulated data as our
 * projection stack for the reconstruction. The reconstruction consists of
 * three Landweber iterations.
 */
void reconstruct(bulk::world& world,
                 const tpt::grcb::node<float>& root,
                 tpt::geometry::base<3_D, float>& g,
                 tpt::volume<3_D, float> v)
{
    auto report = bulk::util::table("Results", "iteration");
    report.columns("FP", "comm", "BP", "total");

    auto s = world.rank();
    auto p = world.active_processors();

    astra::CLogger::setOutputScreen(2, astra::LOG_DEBUG);
    astraCUDA3d::setGPUIndex(s % 4);

    world.log_once("Preparing communication and local geometry...");

    // Prepare tasks
    auto [gathers, scatters, reduces, meta] =
    tasks(world, g, root, tpt::grcb::corners(v));

    // Construct communication buffers
    auto red_buf = bulk::coarray<float>(world, meta.reduction_size);
    // This holds indices: v (row), theta, u (col)
    auto proj_buf = bulk::coarray<float>(world, meta.projection_size);

    // Convert to a local geometry
    auto g_info = construct_geometry_info(g, p, tpt::grcb::corners(v), root);

    // Make local volume geometry
    auto parts = pleiades::partitioning_to_corners(root, tpt::grcb::corners(v));
    auto [a, b] = tpt::grcb::min_max_cube<float>(parts[s]);
    auto nu = (uint32_t)std::get<0>(g_info.local_shape[s]);
    auto nv = (uint32_t)std::get<1>(g_info.local_shape[s]);
    auto sz = b - a;

    // Compute local number of voxels
    auto nx = (uint32_t)(v.voxels()[0] * (sz[0] / v.physical_lengths()[0]) + 0.5f);
    auto ny = (uint32_t)(v.voxels()[1] * (sz[1] / v.physical_lengths()[1]) + 0.5f);
    auto nz = (uint32_t)(v.voxels()[2] * (sz[2] / v.physical_lengths()[2]) + 0.5f);

    // Compute local voxel origin
    auto nox =
    (uint32_t)(v.voxels()[0] * ((a[0] - v.origin()[0]) / v.physical_lengths()[0]) + 0.5f);
    auto noy =
    (uint32_t)(v.voxels()[1] * ((a[1] - v.origin()[1]) / v.physical_lengths()[1]) + 0.5f);
    auto noz =
    (uint32_t)(v.voxels()[2] * ((a[2] - v.origin()[2]) / v.physical_lengths()[2]) + 0.5f);

    // (Local) number of projections
    auto np = (uint32_t)g.projection_count();

    world.log_once("Making ASTRA objects and simulating phantom...");
    // make ASTRA vol geom
    auto vol_geom =
    astra::CVolumeGeometry3D(nx, ny, nz, a[0], a[1], a[2], b[0], b[1], b[2]);

    // Allocate CPU image buffer
    auto buf = std::vector<float>(nx * ny * nz, 0.0f);
    auto cube =
    std::vector<tpt::math::vec3<uint32_t>>{v.voxels() / 5, 4 * v.voxels() / 5};

    for (auto k = std::max(cube[0][2], noz); k < std::min(cube[1][2], noz + nz); ++k) {
        for (auto j = std::max(cube[0][1], noy); j < std::min(cube[1][1], noy + ny); ++j) {
            for (auto i = std::max(cube[0][0], nox);
                 i < std::min(cube[1][0], nox + nx); ++i) {
                auto idx = (k - noz) * nx * ny + (j - noy) * nx + (i - nox);
                buf[idx] = 1.0f;
            }
        }
    }
    auto phantom_file = std::string("phantom_") + std::to_string(s);
    util::write_raw<float>(phantom_file, buf.data(), buf.size());
    world.log("Wrote phantom file: %s", phantom_file);

    // Allocate GPU memory
    auto D_proj = astraCUDA3d::allocateGPUMemory(nu, np, nv, astraCUDA3d::INIT_ZERO);
    assert(D_proj);
    auto D_iter = astraCUDA3d::allocateGPUMemory(nx, ny, nz, astraCUDA3d::INIT_ZERO);
    assert(D_iter);

    astraCUDA3d::SSubDimensions3D dims_vol{nx, ny, nz, nx, nx, ny, nz, 0, 0, 0};
    astraCUDA3d::SSubDimensions3D dims_proj{nu, np, nv, nu, nu,
                                            np, nv, 0,  0,  0};

    astra::CProjectionGeometry3D* proj_geom = util::get_astra_subgeometry(g, g_info, s);

    // Forward project and create 'b',
    auto copy_result = astraCUDA3d::copyToGPUMemory(buf.data(), D_iter, dims_vol);
    auto fp_result = astraCUDA3d::FP(proj_geom, D_proj, &vol_geom, D_iter, 1.0f,
                                     astraCUDA3d::ker3d_default);
    auto copy_back_result =
    astraCUDA3d::copyFromGPUMemory(proj_buf.begin(), D_proj, dims_proj);
    astraCUDA3d::zeroGPUMemory(D_iter, nx, ny, nz);

    if (!copy_result) {
        world.log("Copy -> GPU error");
    }
    if (!fp_result) {
        world.log("FP error");
    }
    if (!copy_back_result) {
        world.log("Copy -> CPU error");
    }

    // Perform a communication step to construct phantom
    gather(red_buf, gathers, proj_buf.begin());
    reduce(red_buf, reduces, proj_buf.begin());
    scatter(proj_buf, scatters, proj_buf.begin());

    auto sino_file = fmt::format("sino_{}_{}_{}_{}", nu, np, nv, s);
    util::write_raw<float>(sino_file, proj_buf.begin(), proj_buf.size());
    world.log("Wrote sinogram to file: %s", sino_file);

    // Store as right-hand side
    auto rhs = std::vector<float>(proj_buf.size());
    std::copy(proj_buf.begin(), proj_buf.end(), rhs.begin());

    world.log_once("Begin reconstruction task...");
    auto iter_timer = bulk::util::timer();
    auto num_iters = 3u;
    for (auto iter = 0u; iter < num_iters; ++iter) {
        auto dt = bulk::util::timer();

        world.log_once("Iteration: %i", iter);

        // ASTRA FP
        astraCUDA3d::zeroGPUMemory(D_proj, nu, np, nv);
        astraCUDA3d::FP(proj_geom, D_proj, &vol_geom, D_iter, 1, astraCUDA3d::ker3d_default);

        auto t_fp = dt.get<std::ratio<1>>();

        // Download from GPU
        astraCUDA3d::copyFromGPUMemory(proj_buf.begin(), D_proj, dims_proj);

        // Communicate
        gather(red_buf, gathers, proj_buf.begin());
        reduce(red_buf, reduces, proj_buf.begin());
        scatter(proj_buf, scatters, proj_buf.begin());

        auto t_comm = dt.get<std::ratio<1>>() - t_fp;

        // Subtract from right-hand side
        for (auto i = 0u; i < rhs.size(); ++i) {
            proj_buf[i] = rhs[i] - proj_buf[i];
        }

        // Copy back to GPU and perform BP
        astraCUDA3d::copyToGPUMemory(proj_buf.begin(), D_proj, dims_proj);
        astraCUDA3d::BP(proj_geom, D_proj, &vol_geom, D_iter, 1, false);

        // Store timings
        auto t_bp = dt.get<std::ratio<1>>() - t_comm - t_fp;
        auto t_total = dt.get<std::ratio<1>>();
        report.row(std::to_string(iter), t_fp, t_comm, t_bp, t_total);
    }
    world.sync();
    auto t_3iter = iter_timer.get<std::ratio<1>>();

    // Store iterate on CPU
    astraCUDA3d::copyFromGPUMemory(buf.data(), D_iter, dims_vol);

    auto recon_file = fmt::format("recon_{}_{}_{}_{}", nx, ny, nz, s);
    util::write_raw<float>(recon_file, buf.data(), buf.size());
    world.log("Wrote reconstruction file: %s", recon_file);

    if (s == 0) {
        std::cout << report.print() << "\n";
        std::cout << "Total for 3 iters: " << t_3iter << " sec.\n";
    }

    astraCUDA3d::freeGPUMemory(D_proj);
    astraCUDA3d::freeGPUMemory(D_iter);
    delete proj_geom;
}

} // namespace pleiades
