/*
 * TODO Global to local indices
 * TODO Remote indices per scanline for each target
 */

#include <random>
#include <vector>

#include <bulk/bulk.hpp>

namespace pleiades {

// a scanlines is a consecutive subarray from position `begin` up to `begin + count`
struct scanline {
    int begin;
    int count;
};

// a face has a list of contributing processors, and a list of scanlines
struct face {
    std::vector<int> contributors;
    std::vector<scanline> scanlines;
};

// for the gather step, one of the contributors of a face is considered its
// owner. the scanlines in `lines` are to be communicated to the owner's
// 'reduction buffer' at the index indicated by the first component of the pair
struct gather_task {
    int owner;
    std::vector<std::pair<int, scanline>> lines;
};

// for the scatter step, the owner communicates the reduction result back to the
// contributors. Each line has an associated tag, which is a list of remote
// indices of the contributors in their main data buffer
struct scatter_task {
    std::vector<int> contributors;
    std::vector<std::pair<std::vector<int>, scanline>> lines;
};

template <typename T>
void gather(bulk::coarray<T> buffer, std::vector<gather_task> tasks, const T* data)
{
    for (auto task : tasks) {
        for (auto [remote, line] : task.lines) {
            auto [begin, count] = line;
            buffer(task.owner)[{remote, remote + count}] = {&data[begin], count};
        }
    }
    buffer.world.sync();
}

template <typename T>
void scatter(bulk::coarray<T> buffer, std::vector<scatter_task> tasks, const T* data)
{
    for (auto task : tasks) {
        for (auto [remotes, line] : task.lines) {
            for (int i = 0; i < task.contributors.size(); ++i) {
                auto remote = remotes[i];
                buffer(task.contributors[i])[{remote, remote + count}] = {&data[begin], count};
            }
        }
        buffer.world.sync();
    }
}

template <typename T>
std::pair<std::vector<gather_task>, std::vector<scatter_task>>
tasks(bulk::world& world,
      const tpt::geometry::base<3_D, T>& g,
      const tpt::grcb::node<T>& root,
      tpt::grcb::cube<T> v)
{
    // maybe best if projections are treated round robin, and independently a
    // collection of gather--scatter tasks are made. Then, we collect these task

    // main question for now: signature of index conversion methods

    // given a collection of scanlines, with global indices in the projection
    // stack, convert to 1) buffer indices and 2) local indices

    // TODO who is responsible for deciding owners of faces
    // TODO round robin stuff

    auto rd = std::random_device();
    auto engine = std::mt19937(rd());

    auto s = world.rank();
    auto p = world.active_processors();

    auto scatters = std::vector<std::vector<scatter_tasks>>(p);
    auto gathers = std::vector<std::vector<gather_tasks>>(p);

    for (int i = s; i < g.projection_count(); i += p) {
        auto pi = g.get_projection(i);

        // we are responsible for the i-th projection
        auto overlay = get_overlay_for_projection(pi, root, v);
        auto faces = get_faces(g, overlay);

        for (auto& face : faces) {
            // assign a random owner to the face off the list of processors
            auto owner = face.contributors[engine() % face.contributors.size()];
            // prepare gather_tasks for contributors, scatter_tasks for owner
            for (auto t : face.contributors) {
                scatters[t].push_back(gather_task{..});
                scatters[t].push_back(scatter_task{..});
            }
        }
    }

    auto sq = bulk::queue<scatter_task>(world);

    for (int t = 0; t < p; ++t) {
        for (auto task : scatters) {
            sq(t).send(task)
        }
    }

    world.sync();
}

} // namespace pleiades
