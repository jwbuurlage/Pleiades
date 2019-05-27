#pragma once

#include <vector>

namespace pleiades {

// I assume pixel coordinates are in 'matrix access form'
struct geometry_info {
    // the number of projections in the geometry
    int projection_count;

    // the shape of the global projections
    // row, column
    std::pair<int, int> shape;

    // corner[t][i]: the upper left corner of projection i on processor t
    // row, column
    std::vector<std::vector<std::pair<int, int>>> corner;

    // columns[t]: the number of columns in the projections on processor t
    std::vector<std::pair<int, int>> local_shape;

    // offsets[t][i]: the offset of the i-th projection in the projection buffer
    // on processor t
    std::vector<std::vector<std::size_t>> offsets;
};

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
