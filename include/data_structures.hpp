#pragma once

#include <cassert>
#include <vector>

namespace pleiades {

struct projection_bboxes {
    // corner_tl[t]: the upper left corner of projection on processor t
    // shape[t]: the shape of the projections on processor t
    // row, column
    std::vector<std::pair<int, int>> corner;
    std::vector<std::pair<int, int>> shape;
};

// I assume pixel coordinates are in 'matrix access form'
struct geometry_info {
    // the number of projections in the geometry
    int projection_count;

    // the shape of the global projections
    // cols x rows
    std::pair<int, int> shape;

    // corner[t][i]: the upper left corner of projection i on processor t
    // col, row
    std::vector<std::vector<std::pair<int, int>>> corner;

    // local_shape[t]: the shape of the projections on processor t
    // this is cols x rows
    // detector shape in TPT is rows x cols
    std::vector<std::pair<int, int>> local_shape;

    // offsets[t][i]: the offset of the i-th projection in the projection buffer
    // on processor t
    std::vector<std::vector<std::size_t>> offsets;
};

// Convert a global index to a local one
// t: target rank
std::size_t localize(geometry_info g, int t, int projection_index, std::size_t pixel)
{
    // global pixel
    // col, row
    auto i = pixel % std::get<0>(g.shape);
    auto j = pixel / std::get<0>(g.shape);

    // local pixel
    // (a, b) = (col, row)
    auto a = i - std::get<0>(g.corner[t][projection_index]);
    auto b = j - std::get<1>(g.corner[t][projection_index]);

    assert(i >= (uint32_t)std::get<0>(g.corner[t][projection_index]));
    assert(j >= (uint32_t)std::get<1>(g.corner[t][projection_index]));
    assert(a < (uint32_t)std::get<0>(g.local_shape[t]));
    if (b >= (uint32_t)std::get<1>(g.local_shape[t])) {
        std::cerr << "ERROR: " << b << " "
                  << (uint32_t)std::get<1>(g.local_shape[t]) << "\n";
        assert(false);
    }

    // local coordinate
    return g.offsets[t][projection_index] +
           b * std::get<0>(g.local_shape[t]) * g.projection_count + a;
}

// scanline is a consecutive subarray from position `begin` up to `begin +
// count`
struct scanline {
    std::size_t begin;
    std::size_t count;
};

// a face has a list of contributing processors, and a list of scanlines
struct face {
    std::vector<int> contributors;
    std::vector<scanline> scanlines;
};

} // namespace pleiades
