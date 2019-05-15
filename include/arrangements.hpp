#include <vector>

// A face overlay of two arrangements with extended face records.
#include <CGAL/Arr_default_overlay_traits.h>
#include <CGAL/Arr_extended_dcel.h>
#include <CGAL/Arr_overlay_2.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Cartesian.h>
#include <CGAL/intersections.h>

#include <boost/geometry.hpp>
namespace bg = boost::geometry;

namespace pleiades {

using tag = std::vector<int>;
struct concatenate_tag {
    tag operator()(const tag& lhs, const tag& rhs) const {
        auto result = lhs;
        for (auto x : rhs) {
            result.push_back(x);
        }
        return result;
    }
};

template <typename T>
void find_part_and_add(const tpt::grcb::node<T>& partitioning,
                       tpt::grcb::cube<T> v,
                       std::vector<tpt::grcb::cube<T>>& result) {
    auto [vl, vr] = tpt::grcb::split_at(v, partitioning.c, partitioning.a);
    if (!partitioning.left) {
        result.push_back(vl);
        result.push_back(vr);
    } else {
        find_part_and_add(*partitioning.left, vl, result);
        find_part_and_add(*partitioning.right, vr, result);
    }
}

template <typename T>
std::vector<tpt::grcb::cube<T>>
partitioning_to_corners(const tpt::grcb::node<T>& partitioning,
                        tpt::grcb::cube<T> v) {
    std::vector<tpt::grcb::cube<T>> result;
    find_part_and_add(partitioning, v, result);
    return result;
}

// using Kernel = CGAL::Cartesian<double>;
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Traits_2 = CGAL::Arr_segment_traits_2<Kernel>;
using Point_2 = Traits_2::Point_2;
using Segment_2 = Traits_2::X_monotone_curve_2;
using Dcel = CGAL::Arr_face_extended_dcel<Traits_2, tag>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2, Dcel>;
using Overlay_traits =
    CGAL::Arr_face_overlay_traits<Arrangement_2, Arrangement_2, Arrangement_2,
                                  concatenate_tag>;
using arrangement = Arrangement_2;

arrangement merge(const arrangement& lhs, const arrangement& rhs) {
    arrangement overlay;
    Overlay_traits overlay_traits;
    CGAL::overlay(lhs, rhs, overlay, overlay_traits);
    return overlay;
}

template <typename T>
auto get_arrangement(tpt::geometry::projection<3_D, T> pi,
                     tpt::grcb::cube<T> corners, int processor_id) {
    // convex hull
    auto [points, hull] = tpt::grcb::shadow(pi, corners);

    // convert hull to arrangement
    arrangement shadow;
    boost::geometry::for_each_segment(hull.outer(), [&](auto s) {
        CGAL::insert_non_intersecting_curve(
            shadow, {{bg::get<0, 0>(s), bg::get<0, 1>(s)},
                     {bg::get<1, 0>(s), bg::get<1, 1>(s)}});
    });

    CGAL_assertion(shadow.number_of_faces() == 2);
    for (auto fit = shadow.faces_begin(); fit != shadow.faces_end(); ++fit) {
        if (fit == shadow.unbounded_face()) {
            continue;
        }
        fit->set_data({processor_id});
    }

    return shadow;
}

template <typename T>
arrangement get_overlay_for_projection(tpt::geometry::projection<3_D, T> pi,
                                       const tpt::grcb::node<T>& root,
                                       tpt::grcb::cube<T> v) {
    auto parts = pleiades::partitioning_to_corners(root, v);
    auto arrangements = std::vector<pleiades::arrangement>(parts.size());

    int s = 0;
    std::transform(parts.begin(), parts.end(), arrangements.begin(),
                   [&](auto corners) {
                       return pleiades::get_arrangement(pi, corners, s++);
                   });

    return std::accumulate(arrangements.begin() + 1, arrangements.end(),
                           arrangements[0], pleiades::merge);
}

// plot the arrangements on a svg, darker colors for more contributors
// convert face to boost geometry polygons
// color number of contributors to the face
template <typename T>
void plot_arrangement(tpt::geometry::projection<3_D, T> pi, std::string name,
                      arrangement overlay) {
    std::ofstream svg("communication_" + name + ".svg");
    auto mapper = bg::svg_mapper<tpt::math::vec2<T>, false>(svg, 500, 500);
    auto det = tpt::grcb::detector(pi);
    mapper.add(det);
    mapper.map(det, "fill-opacity:1.0;fill:rgb(255,255,255);stroke:rgb(0,0,0);"
                    "stroke-width:2");

    for (auto fit = overlay.faces_begin(); fit != overlay.faces_end(); ++fit) {
        if (fit->data().size() < 2) {
            continue;
        } else {
            std::cout << "Face found for: [";
            auto sep = "";
            for (auto t : fit->data()) {
                std::cout << sep << t;
                sep = ", ";
            }
            std::cout << "]\n";

            // plot on svg
            auto face = bg::model::polygon<tpt::math::vec2<T>>();
            auto curr = fit->outer_ccb();
            do {
                auto& x = curr->source()->point();
                bg::append(face.outer(),
                           tpt::math::vec2<T>{x.x().exact().convert_to<T>(),
                                              x.y().exact().convert_to<T>()});
                ++curr;
            } while (curr != fit->outer_ccb());

            mapper.add(face);
            mapper.map(face, "fill-opacity:" +
                                 std::to_string(fit->data().size() / 10.0) +
                                 ";fill:rgb(0,0,0);"
                                 "stroke:rgb(0, 0, 0);stroke-width:2");
        }
    }
}


typedef Kernel::Point_2 Point2;
typedef Kernel::Segment_2 Seg2;
typedef Kernel::Line_2 Line2;

// TEMPORARY: include communication_structures later
struct scanline {
    int begin;
    int count;
};

struct face {
    std::vector<int> contributors;
    std::vector<scanline> scanlines;
};
// ---


template <typename T>
std::vector<pleiades::face> compute_scanlines(tpt::geometry::projection<3_D, T> pi, arrangement overlay) {

    // ASSUMPTIONS:
    //   detector coordinate space:
    //     detector centered around origin
    //     total dimensions given by pi.detector_size
    //     number of pixels given by is pi.detector_shape
    //     order of coordinates: u, v
    //     scanlines have constant v
    //
    //   arrangement:
    //     all shadows can be rounded down up to 1 pixel horizontally, and down up to half a pixel vertically
    //     without dropping required pixels from nodes
    //
    //
    // OUTPUT:
    //   scanline begin index: iv * detector_shape[0] + iu (GLOBAL detector coordinates)

    std::vector<face> result;

    Kernel::FT eds_u(pi.detector_size[0]);
    Kernel::FT eds_v(pi.detector_size[1]);

    std::cout << pi.detector_size[0] << "," << pi.detector_size[1] << std::endl;

    std::vector<int> TEST;
    TEST.resize(pi.detector_shape[0] * pi.detector_shape[1]);

    for (auto fit = overlay.faces_begin(); fit != overlay.faces_end(); ++fit) {
        // skip "outer" face
        if (!fit->has_outer_ccb())
            continue;

        std::cout << "Face: [";
        auto sep = "";
        for (auto t : fit->data()) {
            std::cout << sep << t;
            sep = ", ";
        }
        std::cout << "]\n";

        // Assumption: faces have no holes.
        // This assumption can be dropped, but the algorithm below then also has to iterate over the holes
        assert(fit->holes_begin() == fit->holes_end());

        result.push_back(face());
        face& result_f = result.back();

        result_f.contributors = fit->data();

        for (int iv = 0; iv < pi.detector_shape[1]; ++iv) {
            Kernel::FT v = -eds_v / 2 + eds_v * ( Kernel::FT(2*iv+1) / (2 * pi.detector_shape[1]) );

            Point2 a(-eds_u, v);
            Point2 b(eds_u, v);
            Line2 line(a, b);

            std::vector<Kernel::FT> us;

            auto edge = fit->outer_ccb();
            do {
                Seg2 seg(edge->source()->point(), edge->target()->point());

                auto result = CGAL::intersection(seg, line);
                if (result) {
                    // The intersection is either a point or a segment.

                    // To resolve ambiguity where we exactly hit an endpoint
                    // of an edge (or both of them), we consider the intersection line to
                    // be infinitesimally lower than v.

                    // The implication is that horizontal edges don't intersect the line,
                    // and an endpoint of an edge is hit only if it is the
                    // endpoint with the highest v coordinate.
                    if (const Point2* pp = boost::get<Point2>(&*result)) {
                        Kernel::FT u = pp->x();
                        Kernel::FT ui = (u + eds_u / 2) / eds_u * pi.detector_shape[0];
                        if (*pp == edge->source()->point() ||
                            *pp == edge->target()->point())
                        {
                            Kernel::FT v1 = edge->source()->point().y();
                            Kernel::FT v2 = edge->target()->point().y();
                            Kernel::FT vp = pp->y();
                            assert(v1 != v2);
                            if (vp == CGAL::max(v1, v2))
                                us.push_back(ui);
                        } else
                            us.push_back(ui);
                    } else {
                        assert(boost::get<Seg2>(&*result));
                    }
                }

                ++edge;
            } while (edge != fit->outer_ccb());


            if (us.empty())
                continue;

            std::cout << "Scanline " << iv << " at " << v << ": ";

            assert(us.size() % 2 == 0);

            std::sort(us.begin(), us.end());

            for (unsigned int i = 0; i < us.size() / 2; ++i) {
                Kernel::FT u1i = us[2*i];
                Kernel::FT u2i = us[2*i+1];

                std::cout << u1i << "," << u2i << "; ";

                if (u2i < 0 || u1i >= pi.detector_shape[0])
                    continue;

                if (u1i < 0)
                    u1i = 0;
                if (u2i >= pi.detector_shape[0])
                    u2i = pi.detector_shape[0];

                // convert_to<int> rounds towards zero
                int u1r = u1i.exact().convert_to<int>();
                int u2r = u2i.exact().convert_to<int>();

                if (u1r == u2r)
                    continue;

                assert(u1r >= 0 && u1r <= pi.detector_shape[0] - 1);
                assert(u2r >= 1 && u2r <= pi.detector_shape[0]);
                assert(u2r > u1r);

                int begin = iv * pi.detector_shape[0] + u1r;
                int count = u2r - u1r;

                for (int j = 0; j < count; ++j)
                    TEST[begin + j] += 1;

                std::cout << u1r << "," << count << "; ";

                result_f.scanlines.push_back({ begin, count });

            }
            std::cout << std::endl;
        }
    }

    // Output number of scanlines that overlap each pixel, and do a quick H-convexity check
    for (int y = 0; y < pi.detector_shape[1]; ++y) {
        int state = 0;
        for (int x = 0; x < pi.detector_shape[0]; ++x) {
            int t = TEST[y * pi.detector_shape[0] + x];
            if (state == 0) { // start of row
                assert(t == 0 || t == 1);
                if (t == 1)
                    state = 1;
            } else if (state == 1) { // middle of row
                assert(t == 0 || t == 1);
                if (t == 0)
                    state = 2;
            } else if (state == 2) { // end of row
                assert(t == 0);
            }
            std::cout << t << " ";
        }
        std::cout << std::endl;
    }

    return result;
}

} // namespace pleiades
