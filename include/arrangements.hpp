#include <vector>

// A face overlay of two arrangements with extended face records.
#include <CGAL/Arr_default_overlay_traits.h>
#include <CGAL/Arr_extended_dcel.h>
#include <CGAL/Arr_overlay_2.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Cartesian.h>

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

} // namespace pleiades
