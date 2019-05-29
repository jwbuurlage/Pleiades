#include "fmt/format.h"

#include "tpt/tpt.hpp"
#include <CLI/CLI.hpp>

#include "arrangements.hpp"

int main(int argc, char* argv[])
{
    using T = float;

    CLI::App app{"Restricted geometry stats"};

    int p = 2;
    std::string geometry_filename = "";

    // number of parts
    app.add_option("-p", p, "number of parts", p);

    // file input / output
    app.add_option("--geometry", geometry_filename, "geometry as .toml")->required();

    CLI11_PARSE(app, argc, argv);

    // read problem
    auto problem = tpt::read_configuration<3_D, T>(geometry_filename);
    auto& g = *problem.acquisition_geometry;
    auto v = problem.object_volume;

    auto root = tpt::grcb::partition(v, g, std::log2(p), {});

    auto parts = pleiades::partitioning_to_corners<T>(*root, tpt::grcb::corners(v));

    for (int s = 0; s < p; ++s) {
        auto max_w = 0;
        auto max_h = 0;
        auto sum_pix = 0;

        auto [a, b] = tpt::grcb::min_max_cube<T>(parts[s]);
        auto lv = tpt::volume<3_D, T>({1024, 1024, 1024}, a, b - a);

        // WARNING: unsafe cast to trajectory geometry
        auto rg =
        tpt::distributed::restricted_geometry<T>((tpt::geometry::trajectory<3_D, T>&)g, lv);

        for (auto i = 0; i < rg.projection_count(); ++i) {
            auto shape = rg.projection_shape(i);
            max_w = std::max(max_w, shape[0]);
            max_h = std::max(max_h, shape[1]);
            sum_pix += shape[0] * shape[1];
        }

        auto buffer_pix = max_w * max_h;
        auto avg = (T)(sum_pix / rg.projection_count());

        fmt::print("{} ({}): {} x {} = {} / {} = {}\n", geometry_filename, s,
                   max_w, max_h, buffer_pix, avg, buffer_pix / avg);
    }
}
