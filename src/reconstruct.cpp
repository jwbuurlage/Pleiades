#include "fmt/format.h"

#include "tpt/tpt.hpp"
#include <CLI/CLI.hpp>

#include "reconstruct.hpp"

int main(int argc, char* argv[]) {
    using T = float;

    auto env = bulk::mpi::environment();
    env.spawn(env.available_processors(), [&](auto& world) {
        auto app = CLI::App{"Run a reconstruction job"};

        auto p = 2;
        auto geometry_filename = std::string("");

        // number of parts
        app.add_option("-p", p, "number of parts", p);

        // file input / output
        app.add_option("--geometry", geometry_filename, "geometry as .toml")
            ->required();

        CLI11_PARSE(app, argc, argv);

        // read problem
        auto problem = tpt::read_configuration<3_D, T>(geometry_filename);
        auto& g = *problem.acquisition_geometry;
        auto v = problem.object_volume;

        auto root = tpt::grcb::partition(v, g, std::log2(p), {});

        auto parts =
            pleiades::partitioning_to_corners<T>(*root, tpt::grcb::corners(v));

        pleiades::reconstruct(world, ...);
    });
}
