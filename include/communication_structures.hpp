#include <random>
#include <vector>

#include <bulk/bulk.hpp>

namespace pleiades {

struct scanline {
    int begin;
    int count;
};

struct face {
    std::vector<int> contributors;
    std::vector<scanline> scanlines;
};

struct gather_task {
    int owner;
    std::vector<scanline> local_scanlines;
};

struct scatter_task {
    std::vector<int> contributors;
    std::vector<std::vector<int>> remote;
    std::vector<scanline> local_scanlines;
};

template <typename T>
std::pair<std::vector<gather_task>, std::vector<scatter_task>>
tasks(bulk::world& world, const tpt::geometry::base<3_D, T>& g,
      const tpt::grcb::node<T>& root, tpt::grcb::cube<T> v) {
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

// maybe best if projections are treated round robin, and independently a
// collection of gather--scatter tasks are made. Then, we collect these task

// main question for now: signature of index conversion methods

// given a collection of scanlines, with global indices in the projection
// stack, convert to 1) buffer indices and 2) local indices

// TODO who is responsible for deciding owners of faces
// TODO round robin stuff
} // namespace pleiades

template <typename T>
void gather(bulk::coarray<T> buffer, std::vector<gather_task> tasks,
            const T* data) {
    for (auto task : tasks) {
        auto [t, i] = task.owner;
        for (auto [begin, count] : local_scanlines) {
            buffer(t)[{i, i + count}] = {&data[begin], count};
        }
    }
    buffer.world.sync();
}

template <typename T>
void scatter(bulk::coarray<T> buffer, std::vector<scatter_task> tasks,
             const T* data) {
    for (auto task : tasks) {
        for (auto [t, i] : task.contributors) {
            for (auto [begin, count] : local_scanlines) {
                buffer(t)[{i, i + count}] = {&data[begin], count};
            }
        }
    }
    buffer.world.sync();
}

} // namespace pleiades
