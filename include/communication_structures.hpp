#pragma once

#include <functional>
#include <random>
#include <vector>

#include <bulk/bulk.hpp>

#include "arrangements.hpp"
#include "data_structures.hpp"

namespace pleiades {

// The communication buffers:
// - Buffer (1): projection data buffer on CPU
// - Buffer (2): reduction buffer
// (1) is ordered just like the projection data on the GPU
// (2) is structured in blocks; each block correspond to tasks originating from
// a single processor that chose the local processor as an owner. Each scanline
// reduction input is consecutive in this block (count * #contributors)

// The communication procedure:
// A. GPU         --- DOWNLOAD --> Buffer (1)
// B. Buffer (1)  ---  GATHER  --> Buffer (2)
// C. Buffer (2)  ---  REDUCE  --> Buffer (1)
// D. Buffer (1)  ---  SCATTER --> Buffer (1)
// E. Buffer (1)  ---  UPLOAD  --> GPU

// TASKS:
// There is one gather tasks for each contributor of a face
// There is one scatter tasks for the owner of a face

// for the gather step, one of the contributors of a face is considered its
// owner. the scanlines in `lines` are to be communicated to the owner's
// 'reduction buffer' at the index indicated by the first component of the pair
struct gather_task {
    int owner;
    std::vector<std::pair<std::size_t, scanline>> lines;
};

// for the scatter step, the owner communicates the reduction result back to the
// contributors. Each line has an associated tag, which is a list of remote
// indices of the contributors in their main data buffer
// TODO here the scanline indices are in the gather result buffer
struct scatter_task {
    std::vector<int> contributors;
    std::vector<std::pair<std::vector<std::size_t>, scanline>> lines;
};

// TODO implement gather and scatter steps
// template <typename T>
// void gather(bulk::coarray<T> buffer, std::vector<gather_task> tasks,
//            const T* data) {
//    for (auto task : tasks) {
//        for (auto [remote, line] : task.lines) {
//            auto [begin, count] = line;
//            buffer(task.owner)[{remote, remote + count}] = {&data[begin],
//                                                            count};
//        }
//    }
//    buffer.world.sync();
//}
//
// template <typename T>
// void scatter(bulk::coarray<T> buffer, std::vector<scatter_task> tasks,
//             const T* data) {
//    for (auto task : tasks) {
//        for (auto [remotes, line] : task.lines) {
//            for (auto i = 0u; i < task.contributors.size(); ++i) {
//                auto remote = remotes[i];
//                buffer(task.contributors[i])[{remote, remote + count}] = {
//                    &data[begin], count};
//            }
//        }
//        buffer.world.sync();
//    }
//}

/** Outputs the gather and scatter tasks for the local processor */
template <typename T>
std::pair<std::vector<gather_task>, std::vector<scatter_task>>
tasks(bulk::world& world,
      const tpt::geometry::base<3_D, T>& acquisition_geometry,
      const tpt::grcb::node<T>& root, tpt::grcb::cube<T> v) {
    // TODO construct_geometry_info(acquisition_geometry);
    auto g_info = geometry_info{};

    // alias local rank and number of processors
    auto s = world.rank();
    auto p = world.active_processors();

    // The strategy is as follows.
    // 1. Assign the projections round robin, and treat them independently. For
    // each projection, gather and scatter tasks are constructed with the
    // correct indices. This happens in two phases.
    //   A) Owners are assigned to the faces, and the buffers are measured for
    //   size B) The tasks are generated, with correct indices
    // 2. For all the projections that are being processed by the local rank,
    // gather the task together in lists (`scatters`, `gathers`). These tasks
    // are grouped by responsible rank.
    // 3. Communicate the gather and scatter tasks using a queue to the
    // responsible ranks.
    // 4. Read out the queue, and gather the tasks in a vector.
    //
    // The gather tasks correspond to a reduction operations, that add up the
    // local scanlines. We assume that the result of this is written into the
    // main 'projection data buffer', and the scatter operation also directly
    // writes to this 'projection data buffer'. This buffer should be on the
    // CPU, and registered using Bulk to enable communication optimizations.

    // owners are assigned randomly, so we construct an engine.
    auto rd = std::random_device();
    auto engine = std::mt19937(rd());

    // cache faces of local projections, and the assigned owners
    auto faces = std::vector<std::vector<face>>();
    auto owners = std::vector<std::vector<int>>();

    // the buffer size that we require for each processor
    auto B = std::vector<std::size_t>(p, 0);

    // PHASE A: 'Dry run': assign owners, and measure buffers
    // process projections in a round-robin fashion
    // Here, i is the local index, and proj_id is the global index
    // The only goal here is to compute D
    for (auto i = 0, proj_id = s;
         proj_id < acquisition_geometry.projection_count();
         proj_id += p, i += 1) {
        auto pi = acquisition_geometry.get_projection(proj_id);

        // get faces for the i-th projection
        auto overlay = get_overlay_for_projection(pi, root, v);
        faces.push_back(compute_scanlines(pi, overlay));

        // make zero-initialized list of owners
        owners.push_back(std::vector<int>(0, faces[i].size()));

        // We need to:
        // - assign and cache the owner
        // - count the total buffer size
        auto f = 0;
        for (auto& face : faces[i]) {
            // assign a random owner to the face from the list of contributors
            owners[i][f] =
                face.contributors[engine() % face.contributors.size()];

            // measure the number of pixels
            auto pixels = std::accumulate(
                face.scanlines.begin(), face.scanlines.end(), 0u,
                [](auto total, auto l) { return total + l.count; });

            B[owners[i][f]] += face.contributors.size() * pixels;

            f += 1;
        }
    }

    // now we have B[t], the local buffer offsets can be computed this is a full
    // p^2-relation, after this communication step we can compute partial sums.
    auto C = bulk::coarray<std::size_t>(world, p * p);
    for (auto u = 0; u < p; ++u) {
        for (auto t = 0; t < p; ++t) {
            C(u)[t * p + s] = B[t];
        }
    }
    world.sync();

    // D[t] is the beginning of our gather buffer (Buffer (2)) portion in
    // processor t
    auto D = std::vector<int>(p, 0);
    for (int t = 0; t < p; ++t) {
        for (int u = 0; u < s; ++u) {
            D[t] += C[t * p + u];
        }
    }

    // PHASE B: Construct task info
    // prepare gather_tasks for contributors, scatter_tasks for owner
    // GOAL:
    // For each face:
    // - construct information for #contributors gather tasks,
    // - construct information for a scatter task
    // This information is:
    // - gather task:
    // {contributor, owner, [... { remote_begin, {local_begin, count} } ...]}
    // - scatter task:
    // {contributors, owner, projection_id, [... { global_begin, count } ...]}

    // we construct a single task, that makes it easy to send non-localized
    // tasks. We localize at the receive site.
    // tasks generated by the processing of the local projections.

    struct scatter_info {
        std::vector<int> contributors;
        int projection_id;
        std::vector<scanline> lines;
    };

    struct gather_info {
        int owner;
        std::vector<std::pair<std::size_t, scanline>> lines;
    };

    auto ss = std::vector<std::vector<scatter_info>>(p);
    auto gs = std::vector<std::vector<gather_info>>(p);

    // juggle indices
    for (auto i = 0, proj_id = s;
         proj_id < acquisition_geometry.projection_count();
         proj_id += p, i += 1) {

        auto f = 0;
        for (auto& face : faces[i]) {
            auto t = owners[i][f];

            // now we construct the task info
            // first, for each contributor we construct the gather task. the
            // lines are to be filled
            auto gathers =
                std::vector<gather_info>(face.contributors.size(), {t, {}});
            for (auto [begin, count] : face.scanlines) {
                for (auto i = 0u; i < face.contributors.size(); ++i) {
                    auto u = face.contributors[i];
                    gathers[i].lines.push_back(
                        {D[t], {localize(g_info, u, proj_id, begin), count}});
                    D[t] += count;
                }
            }

            for (auto i = 0u; i < face.contributors.size(); ++i) {
                auto u = face.contributors[i];
                gs[u].push_back(gathers[i]);
            }
            ss[t].push_back({face.contributors, proj_id, face.scanlines});

            ++f;
        }
    }

    // PHASE C: Distribute all tasks
    auto sq = bulk::queue<int[], int, scanline[]>(world);
    auto gq = bulk::queue<int, std::pair<std::size_t, scanline>[]>(world);

    for (int t = 0; t < p; ++t) {
        for (auto info : ss[t]) {
            sq(t).send(info.contributors, info.projection_id, info.lines);
        }
        for (auto info : gs[t]) {
            gq(t).send(info.owner, info.lines);
        }
    }

    world.sync();

    // next: construct tasks
    std::vector<gather_task> local_gathers;
    std::vector<scatter_task> local_scatters;
    local_gathers.reserve(gq.size());
    local_scatters.reserve(sq.size());

    for (auto [owner, lines] : gq) {
        local_gathers.push_back({owner, lines});
    }

    for (auto [contributors, proj_id, lines] : sq) {
        auto task = scatter_task{contributors, {}};
        for (auto [begin, count] : lines) {
            std::vector<std::size_t> begins(contributors.size());
            for (auto i = 0u; i < contributors.size(); ++i) {
                auto u = contributors[i];
                begins[i] = localize(g_info, u, proj_id, begin);
            }
            auto local_begin = localize(g_info, s, proj_id, begin);
            task.lines.push_back({begins, {local_begin, count}});
        }
        local_scatters.push_back(task);
    }

    return {local_gathers, local_scatters};
}

} // namespace pleiades
