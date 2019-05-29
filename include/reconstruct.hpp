#include <bulk/bulk.hpp>

#include "communication_structures.hpp"

namespace pleiades {

template <typename T>
void reconstruct(bulk::world& world,
                 const tpt::grcb::node<T>& root,
                 tpt::geometry::base<3_D, T>& g,
                 tpt::volume<3_D, T> v)
{
    // prepare tasks
    auto [gathers, scatters] = tasks(world, g, root, tpt::grcb::corners(v));
}

} // namespace pleiades
