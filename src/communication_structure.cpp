#include <filesystem>
namespace fst = std::filesystem;

#include "tpt/tpt.hpp"
#include <CLI/CLI.hpp>

#include "arrangements.hpp"
#include <boost/geometry.hpp>
namespace bg = boost::geometry;

int main(int argc, char* argv[]) {
    using T = float;

    CLI::App app{"Tests for arrangement communication"};

    int p = 2;
    std::string geometry_filename = "";

    // number of parts
    app.add_option("-p", p, "number of parts", p);

    // file input / output
    app.add_option("--geometry", geometry_filename, "geometry as .toml")
        ->required();

    CLI11_PARSE(app, argc, argv);

    auto name = fst::path(geometry_filename).stem().string();

    // read problem
    auto problem = tpt::read_configuration<3_D, T>(geometry_filename);
    auto& g = *problem.acquisition_geometry;
    auto v = problem.object_volume;

    // take projections, project and plot
    auto corners = tpt::grcb::corners(v);
    auto root = tpt::grcb::partition(v, g, std::log2(p), {});

    // plot overlaps
    for (int i = 0; i < 9; ++i) {
        auto pi = g.get_projection(g.projection_count() / 9 * i);
        tpt::grcb::plot::overlaps(
            pi, root, corners, "full_overlap_" + name + "_" + std::to_string(i),
            p);
    }

    auto pi = g.get_projection(0);
    auto shadows = pleiades::get_shadows_for_projection(pi, *root, corners);
    auto overlay = pleiades::get_overlay_for_projection<T>(shadows);
    pleiades::plot_arrangement(pi, name, overlay);

    return 0;
}
