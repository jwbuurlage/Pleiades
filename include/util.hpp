#include <fstream>
#include <string>

#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/Globals.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/cuda/3d/mem3d.h>


namespace pleiades {

namespace util {

std::string info(const astra::CConeVecProjectionGeometry3D& x)
{
    auto ss = std::stringstream("");

    auto vectors = x.getProjectionVectors();

    ss << "DetectorRowCount: " << x.getDetectorRowCount() << ", ";
    ss << "DetectorColCount: " << x.getDetectorColCount() << ", ";
    ss << "ProjectionCount: " << x.getProjectionCount() << ", ";
    ss << "Vectors: [\n[" << vectors[0].fSrcX << ", " << vectors[0].fSrcY
       << ", " << vectors[0].fSrcZ << "\n"
       << vectors[0].fDetSX << ", " << vectors[0].fDetSY << ", "
       << vectors[0].fDetSZ << " ... "
       << "\n"
       << vectors[0].fDetUX << ", " << vectors[0].fDetUY << ", "
       << vectors[0].fDetUZ << "\n"
       << vectors[0].fDetVX << ", " << vectors[0].fDetVY << ", " << vectors[0].fDetVZ
       << "], [" << vectors[1].fSrcX << ", " << vectors[1].fSrcY << "...]...]";

    return ss.str();
}

std::string info(const astra::CVolumeGeometry3D& x)
{
    auto ss = std::stringstream("");

    ss << "Min: [" << x.getWindowMinX() << ", " << x.getWindowMinY() << ", "
       << x.getWindowMinZ() << "], ";
    ss << "Max: [" << x.getWindowMaxX() << ", " << x.getWindowMaxY() << ", "
       << x.getWindowMaxZ() << "], ";
    ss << "Shape: [" << x.getGridRowCount() << ", " << x.getGridColCount()
       << ", " << x.getGridSliceCount() << "]";

    return ss.str();
}

/**
 * Writes the data in the array `data` of size `count` to the file with
 * extension-less filename `basename` in raw format (binary blob).
 */
template <typename T>
void write_raw(std::string basename, T* data, std::size_t count)
{
    std::ofstream ofile(basename + ".raw", std::ios::binary);
    ofile.write((char*)data, count * sizeof(T));
}

astra::SConeProjection get_astra_vectors(const tpt::geometry::base<3_D, float>& g, int index)
{
    tpt::math::vec<3_D, float> src = g.source_location(index);
    tpt::math::vec<3_D, float> det = g.detector_corner(index);
    std::array<tpt::math::vec<3_D, float>, 2> delta = g.projection_delta(index);

    // Assumption for data ordering:
    // - volume: x, y, z
    // - detector: v, u
    astra::SConeProjection vec;
    vec.fSrcX = src[0];
    vec.fSrcY = src[1];
    vec.fSrcZ = src[2];
    vec.fDetSX = det[0];
    vec.fDetSY = det[1];
    vec.fDetSZ = det[2];
    vec.fDetUX = delta[0][0];
    vec.fDetUY = delta[0][1];
    vec.fDetUZ = delta[0][2];
    vec.fDetVX = delta[1][0];
    vec.fDetVY = delta[1][1];
    vec.fDetVZ = delta[1][2];

    return vec;
}

astra::SConeProjection get_astra_subvectors(const tpt::geometry::base<3_D, float>& g,
                                            int index,
                                            const geometry_info& gi,
                                            int proc_id)
{
    astra::SConeProjection vec = get_astra_vectors(g, index);

    // Translate the detector to the local corner
    vec.fDetSX += std::get<1>(gi.corner[proc_id][index]) * vec.fDetVX;
    vec.fDetSY += std::get<1>(gi.corner[proc_id][index]) * vec.fDetVY;
    vec.fDetSZ += std::get<1>(gi.corner[proc_id][index]) * vec.fDetVZ;

    vec.fDetSX += std::get<0>(gi.corner[proc_id][index]) * vec.fDetUX;
    vec.fDetSY += std::get<0>(gi.corner[proc_id][index]) * vec.fDetUY;
    vec.fDetSZ += std::get<0>(gi.corner[proc_id][index]) * vec.fDetUZ;

    return vec;
}

astra::CConeVecProjectionGeometry3D*
get_astra_subgeometry(const tpt::geometry::base<3_D, float>& g, const geometry_info& gi, int proc)
{
    astra::SConeProjection* vecs = new astra::SConeProjection[gi.projection_count];
    for (auto i = 0; i < gi.projection_count; ++i)
        vecs[i] = get_astra_subvectors(g, i, gi, proc);
    astra::CConeVecProjectionGeometry3D* geom =
    new astra::CConeVecProjectionGeometry3D(gi.projection_count,
                                            std::get<1>(gi.local_shape[proc]),
                                            std::get<0>(gi.local_shape[proc]), vecs);
    delete[] vecs;

    return geom;
}

} // namespace util
} // namespace pleiades
