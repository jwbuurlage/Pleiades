#include <bulk/bulk.hpp>

namespace pleiades {

void reconstruct(bulk::world& world, partitioning root, geometry g, volume v) {
    // prepare tasks
    auto [gathers, scatters] = tasks(world, g, root, v);
}

} // namespace pleiades
