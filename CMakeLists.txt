cmake_minimum_required(VERSION 3.8)
project(pleiades)

include(FindPkgConfig)

add_subdirectory(ext/tpt)
add_subdirectory(ext/bulk)

set(SOURCES "src/pleiades.cpp")

add_library(pleiades ${SOURCES})
target_compile_options(pleiades PUBLIC
    "-Wfatal-errors"
    "-Werror"
    "-Wall"
    "-O3")

# --------------------------------------------------------------------------------------------
# ASTRA
pkg_check_modules(ASTRA REQUIRED "libastra")

add_library(astra-toolbox INTERFACE)
target_include_directories(astra-toolbox SYSTEM INTERFACE ${ASTRA_INCLUDE_DIRS})
target_compile_definitions(astra-toolbox INTERFACE "ASTRA_CUDA")
target_link_libraries(astra-toolbox INTERFACE ${ASTRA_LDFLAGS})
# --------------------------------------------------------------------------------------------

set(CGAL_DIR "ext/cgal/")
set(CGAL_HEADER_ONLY on)
find_package(CGAL)

target_include_directories(pleiades SYSTEM PUBLIC ${CGAL_INCLUDE_DIRS})
target_include_directories(pleiades PUBLIC "include")
target_compile_definitions(pleiades PUBLIC CGAL_HEADER_ONLY)
target_compile_definitions(pleiades PUBLIC CGAL_HEADER_ONLY)

target_link_libraries(pleiades PUBLIC
    ${CGAL_LIBS}
    tpt
    gmp
    mpfr
    astra-toolbox
)

add_executable("comstruct" "src/communication_structure.cpp")
target_link_libraries("comstruct" pleiades)

add_executable("rgstats" "src/restricted_geometry_stats.cpp")
target_link_libraries("rgstats" pleiades)

find_package(MPI)

add_executable("reconstruct" "src/reconstruct.cpp")
target_link_libraries("reconstruct" pleiades MPI::MPI_CXX)

