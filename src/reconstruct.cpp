#include <bulk/backends/mpi/mpi.hpp>
#include <bulk/bulk.hpp>

#include "fmt/format.h"

#include "tpt/tpt.hpp"
#include <CLI/CLI.hpp>

#include "reconstruct.hpp"

using T = float;

int main(int argc, char* argv[])
{
    auto app = CLI::App{"Run a reconstruction job"};

    auto p = 2;
    auto geometry_filename = std::string("");

    // number of parts
    app.add_option("-p", p, "number of parts", p);

    // file input / output
    app.add_option("--geometry", geometry_filename, "geometry as .toml")->required();
    CLI11_PARSE(app, argc, argv);

    auto env = bulk::mpi::environment();
    env.spawn(env.available_processors(), [&](auto& world) {
        world.log("Starting reconstruction for %s", geometry_filename.c_str());

        // read problem
        auto problem = tpt::read_configuration<3_D, T>(geometry_filename);
        auto& g = *problem.acquisition_geometry;
        auto v = problem.object_volume;

        // run reconstruction
        // TODO partitioning should be the same everywhere..
        // 1. load from file
        // 2. (current) midway
        // 3. (preferred?) perform on s = 0 and distribute
        world.log("Partitioning...");
        auto root = tpt::grcb::partition(v, g, std::log2(p), {});
        world.log("Reconstructing...");
        pleiades::reconstruct(world, *root, g, v);
    });

    return 0;
}
