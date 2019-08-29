#include <fstream>

#include <bulk/bulk.hpp>

#include "tpt/math/stringify.hpp"

#include "communication_structures.hpp"

#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/Globals.h>
#include <astra/Logging.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/cuda/3d/mem3d.h>

namespace pleiades {

std::string info(const astra::CConeVecProjectionGeometry3D& x)
{
    auto ss = std::stringstream("");

    auto vectors = x.getProjectionVectors();

    ss << "DetectorRowCount: " << x.getDetectorRowCount() << ", ";
    ss << "DetectorColCount: " << x.getDetectorColCount() << ", ";
    ss << "ProjectionCount: " << x.getProjectionCount() << ", ";
    ss << "Vectors: [\n[" << vectors[0].fSrcX << ", " << vectors[0].fSrcY
       << ", " << vectors[0].fSrcZ << "\n"
       << vectors[0].fDetSX << ", " << vectors[0].fDetSY << ", "
       << vectors[0].fDetSZ << " ... "
       << "\n"
       << vectors[0].fDetUX << ", " << vectors[0].fDetUY << ", "
       << vectors[0].fDetUZ << "\n"
       << vectors[0].fDetVX << ", " << vectors[0].fDetVY << ", " << vectors[0].fDetVZ
       << "], [" << vectors[1].fSrcX << ", " << vectors[1].fSrcY << "...]...]";

    return ss.str();
}

std::string info(const astra::CVolumeGeometry3D& x)
{
    auto ss = std::stringstream("");

    ss << "Min: [" << x.getWindowMinX() << ", " << x.getWindowMinY() << ", "
       << x.getWindowMinZ() << "], ";
    ss << "Max: [" << x.getWindowMaxX() << ", " << x.getWindowMaxY() << ", "
       << x.getWindowMaxZ() << "], ";
    ss << "Shape: [" << x.getGridRowCount() << ", " << x.getGridColCount()
       << ", " << x.getGridSliceCount() << "]";

    return ss.str();
}

// TODO implement gather and scatter steps
template <typename T>
void gather(bulk::coarray<T>& red_buf, const std::vector<gather_task>& tasks, T* proj_data)
{
    for (auto task : tasks) {
        for (auto [remote, line] : task.lines) {
            auto [begin, count] = line;
            // TODO replace indices in bulk with std::size_t...
            assert((remote + count) < ((1u << 31) - 1));

            red_buf(task.owner)[{(int)remote, (int)(remote + count)}] = {&proj_data[begin], count};
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
    red_buf.world().log("Performed %u flops in reduce in %u tasks", flops, tasks.size());
}

template <typename T>
void scatter(bulk::coarray<T>& proj_buf, const std::vector<scatter_task>& tasks, T* proj_data)
{
    for (auto task : tasks) {
        for (auto [begins, line] : task.lines) {
            for (auto i = 0u; i < task.contributors.size(); ++i) {
                auto remote = begins[i];
                auto [local, count] = line;

                assert((remote + count) < ((1u << 31) - 1));

                proj_buf(task.contributors[i])[{(int)remote, (int)(remote + count)}] = {
                &proj_data[local], count};
            }
        }
    }
    proj_buf.world().sync();
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
    vec.fDetUX = delta[0][0];
    vec.fDetUY = delta[0][1];
    vec.fDetUZ = delta[0][2];
    vec.fDetVX = delta[1][0];
    vec.fDetVY = delta[1][1];
    vec.fDetVZ = delta[1][2];

    return vec;
}

astra::SConeProjection get_astra_subvectors(const tpt::geometry::base<3_D, float>& g,
                                            int index,
                                            const geometry_info& gi,
                                            int proc_id)
{
    astra::SConeProjection vec = get_astra_vectors(g, index);

    // translate detector to local corner

    vec.fDetSX += std::get<1>(gi.corner[proc_id][index]) * vec.fDetVX;
    vec.fDetSY += std::get<1>(gi.corner[proc_id][index]) * vec.fDetVY;
    vec.fDetSZ += std::get<1>(gi.corner[proc_id][index]) * vec.fDetVZ;

    vec.fDetSX += std::get<0>(gi.corner[proc_id][index]) * vec.fDetUX;
    vec.fDetSY += std::get<0>(gi.corner[proc_id][index]) * vec.fDetUY;
    vec.fDetSZ += std::get<0>(gi.corner[proc_id][index]) * vec.fDetUZ;

    return vec;
}

astra::CConeVecProjectionGeometry3D*
get_astra_subgeometry(const tpt::geometry::base<3_D, float>& g, const geometry_info& gi, int proc)
{
    astra::SConeProjection* vecs = new astra::SConeProjection[gi.projection_count];
    for (auto i = 0; i < gi.projection_count; ++i)
        vecs[i] = get_astra_subvectors(g, i, gi, proc);
    astra::CConeVecProjectionGeometry3D* geom =
    new astra::CConeVecProjectionGeometry3D(gi.projection_count,
                                            std::get<1>(gi.local_shape[proc]),
                                            std::get<0>(gi.local_shape[proc]), vecs);
    delete[] vecs;

    return geom;
}

template <typename T>
void write_raw(std::string basename, T* data, std::size_t count)
{
    std::ofstream ofile(basename + ".raw", std::ios::binary);
    ofile.write((char*)data, count * sizeof(T));
}

void reconstruct(bulk::world& world,
                 const tpt::grcb::node<float>& root,
                 tpt::geometry::base<3_D, float>& g,
                 tpt::volume<3_D, float> v)
{
    // ... do some Landweber iterations
    auto s = world.rank();
    auto p = world.active_processors();

    astra::CLogger::setOutputScreen(2, astra::LOG_DEBUG);

    world.log("Computing tasks");

    // prepare tasks
    auto [gathers, scatters, reduces, meta] =
    tasks(world, g, root, tpt::grcb::corners(v));

    world.log("#gathers: %d, #scatters: %d, #reduces: %d", gathers.size(),
              scatters.size(), reduces.size());
    world.log("Tasks computed");

    for (auto i = 0u; i < 10; ++i) {
        auto [in, count, contributors, out] = reduces[i];
        world.log("[in %d, count %d, contributors %d, out %d]", in, count, contributors, out);
    }

    // make projection and reduction buffers
    // what size?
    // should `tasks` return additional metadata?
    // (we probably need geometry_info for generating geometries.)

    auto red_buf = bulk::coarray<float>(world, meta.reduction_size);

    // proj_buf has indices:
    // v (row), theta, u (col)
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

    // TODO simplify using something like
    // auto [nu, nv] = g_info.local_shape[s];
    // auto Ns = tpt::math::vec3<float>(v.voxels()) * (sz / v.physical_lengths()) +
    //  tpt::math::vec3<float>(0.5f);
    // TODO ... and a series of rounding ops

    // compute local number of voxels
    auto nx = (uint32_t)(v.voxels()[0] * (sz[0] / v.physical_lengths()[0]) + 0.5f);
    auto ny = (uint32_t)(v.voxels()[1] * (sz[1] / v.physical_lengths()[1]) + 0.5f);
    auto nz = (uint32_t)(v.voxels()[2] * (sz[2] / v.physical_lengths()[2]) + 0.5f);

    // local voxel origin
    auto nox =
    (uint32_t)(v.voxels()[0] * ((a[0] - v.origin()[0]) / v.physical_lengths()[0]) + 0.5f);
    auto noy =
    (uint32_t)(v.voxels()[1] * ((a[1] - v.origin()[1]) / v.physical_lengths()[1]) + 0.5f);
    auto noz =
    (uint32_t)(v.voxels()[2] * ((a[2] - v.origin()[2]) / v.physical_lengths()[2]) + 0.5f);
    auto np = (uint32_t)g.projection_count();

    world.log("Core stats [%d, %d, %d]+[%d, %d, %d], {%d, %d, %d}", nox, noy,
              noz, nx, ny, nz, nu, nv, np);

    world.log("Making ASTRA objects");
    // make ASTRA vol geom
    auto vol_geom =
    astra::CVolumeGeometry3D(nx, ny, nz, a[0], a[1], a[2], b[0], b[1], b[2]);

    world.log("Constructing phantom");
    // Allocate CPU image buffer
    auto buf = std::vector<float>(nx * ny * nz, 0.0f);
    auto cube =
    std::vector<tpt::math::vec3<uint32_t>>{v.voxels() / 5, 4 * v.voxels() / 5};

    world.log("Cube %s %s", tpt::math::to_string<3_D, int>(cube[0]).c_str(),
              tpt::math::to_string<3_D, int>(cube[1]).c_str());

    for (auto k = std::max(cube[0][2], noz); k < std::min(cube[1][2], noz + nz); ++k) {
        for (auto j = std::max(cube[0][1], noy); j < std::min(cube[1][1], noy + ny); ++j) {
            for (auto i = std::max(cube[0][0], nox);
                 i < std::min(cube[1][0], nox + nx); ++i) {
                auto idx = (k - noz) * nx * ny + (j - noy) * nx + (i - nox);
                buf[idx] = 1.0f;
            }
        }
    }
    write_raw<float>(std::string("phantom_") + std::to_string(s), buf.data(),
                     buf.size());

    // TODO forward project and create 'b'

    // collect on s = 0 and plot
    // auto full_image = std::vector<float>(tpt::math::product<3_D, int>(v.voxels()));
    // auto q = bulk::queue<tpt::math::vec3<uint32_t>, tpt::math::vec3<uint32_t>, T[]>(world);
    // write_raw("phantom_gathered", full_image.data(), full_image.size());

    // Allocate GPU memory
    auto D_proj = astraCUDA3d::allocateGPUMemory(nu, np, nv, astraCUDA3d::INIT_ZERO);
    assert(D_proj);
    auto D_iter = astraCUDA3d::allocateGPUMemory(nx, ny, nz, astraCUDA3d::INIT_ZERO);
    assert(D_iter);

    astraCUDA3d::SSubDimensions3D dims_vol{nx, ny, nz, nx, nx, ny, nz, 0, 0, 0};
    astraCUDA3d::SSubDimensions3D dims_proj{nu, np, nv, nu, nu,
                                            np, nv, 0,  0,  0};

    astra::CProjectionGeometry3D* proj_geom = get_astra_subgeometry(g, g_info, s);

    std::cout << info(vol_geom) << "\n";
    // SHAME...! SHAME...! SHAME...!
    std::cout
    << info(*dynamic_cast<astra::CConeVecProjectionGeometry3D*>(proj_geom)) << "\n";

    // forward project and create 'b',
    // TODO use something other than D_proj
    auto copy_result = astraCUDA3d::copyToGPUMemory(buf.data(), D_iter, dims_vol);
    auto fp_result = astraCUDA3d::FP(proj_geom, D_proj, &vol_geom, D_iter, 1.0f,
                                     astraCUDA3d::ker3d_default);
    auto copy_back_result =
    astraCUDA3d::copyFromGPUMemory(proj_buf.begin(), D_proj, dims_proj);

    if (!copy_result) {
        world.log("Copy -> GPU error");
    }
    if (!fp_result) {
        world.log("FP error");
    }
    if (!copy_back_result) {
        world.log("Copy -> CPU error");
    }

    world.log("gather");
    gather(red_buf, gathers, proj_buf.begin());
    world.log("reduce");
    reduce(red_buf, reduces, proj_buf.begin());
    world.log("scatter");
    scatter(proj_buf, scatters, proj_buf.begin());

    world.log("write sino");
    write_raw<float>(fmt::format("sino_{}_{}_{}_{}", nu, np, nv, s),
                     proj_buf.begin(), proj_buf.size());

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


        // (can do inner products in data space now before scatter (for cgls))?
        world.log("scatter");
        scatter(proj_buf, scatters, proj_buf.begin());

        // TODO subtract from b
        // EVERYONE HAS CORRECT FP VALUES FOR ALL VALUES IN PROJ_BUF

        // upload to GPU
        world.log("cpu -> gpu");
        astraCUDA3d::copyToGPUMemory(proj_buf.begin(), D_proj, dims_proj);

        // ASTRA bp (D_proj -> add to D_iter)
        // TODO check last argument
        world.log("bp");
        astraCUDA3d::BP(proj_geom, D_proj, &vol_geom, D_iter, 1, false);
    }

    // Store D_iter
    astraCUDA3d::copyFromGPUMemory(buf.data(), D_iter, dims_vol);


    world.log("writing recon");
    write_raw<float>(std::string("recon_") + std::to_string(s), buf.data(),
                     buf.size());

    astraCUDA3d::freeGPUMemory(D_proj);
    astraCUDA3d::freeGPUMemory(D_iter);
    delete proj_geom;
}

} // namespace pleiades
