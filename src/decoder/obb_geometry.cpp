#include "vie/decoder/obb.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace vie {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kInsideSlack = 1e-3;
constexpr double kIsotropicRelative = 1e-3;

struct Pt {
  double x = 0;
  double y = 0;
};

Pt sub(Pt left, Pt right) { return {left.x - right.x, left.y - right.y}; }

double cross(Pt left, Pt right) { return left.x * right.y - left.y * right.x; }

void require_box(const OrientedBox& box, const char* name) {
  if (!std::isfinite(box.cx) || !std::isfinite(box.cy) || !std::isfinite(box.width) ||
      !std::isfinite(box.height) || !std::isfinite(box.angle)) {
    throw VieError(ErrorCode::InvalidArgument, std::string(name) + " has a non-finite value");
  }
  if (!(box.width > 0.f) || !(box.height > 0.f)) {
    throw VieError(ErrorCode::InvalidArgument, std::string(name) + " width and height must be positive");
  }
}

std::array<Pt, 4> corners_of(const OrientedBox& box) {
  const double cosine = std::cos(static_cast<double>(box.angle));
  const double sine = std::sin(static_cast<double>(box.angle));
  const double half_w = static_cast<double>(box.width) * 0.5;
  const double half_h = static_cast<double>(box.height) * 0.5;
  const Pt along{half_w * cosine, half_w * sine};
  const Pt across{-half_h * sine, half_h * cosine};
  const Pt center{box.cx, box.cy};
  return {Pt{center.x + along.x + across.x, center.y + along.y + across.y},
          Pt{center.x + along.x - across.x, center.y + along.y - across.y},
          Pt{center.x - along.x - across.x, center.y - along.y - across.y},
          Pt{center.x - along.x + across.x, center.y - along.y + across.y}};
}

double signed_area(const std::vector<Pt>& points) {
  double sum = 0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Pt& current = points[index];
    const Pt& next = points[(index + 1) % points.size()];
    sum += cross(current, next);
  }
  return sum * 0.5;
}

std::vector<Pt> ccw_corners(const OrientedBox& box) {
  const std::array<Pt, 4> corners = corners_of(box);
  std::vector<Pt> points(corners.begin(), corners.end());
  if (signed_area(points) < 0) std::reverse(points.begin(), points.end());
  return points;
}

Pt line_hit(Pt start, Pt end, Pt edge_a, Pt edge_b) {
  const Pt se = sub(end, start);
  const Pt ab = sub(edge_b, edge_a);
  const double denom = cross(se, ab);
  if (std::fabs(denom) < 1e-18) return end;
  double t = cross(sub(edge_a, start), ab) / denom;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  return {start.x + se.x * t, start.y + se.y * t};
}

std::vector<Pt> clip_convex(const std::vector<Pt>& subject, const std::vector<Pt>& clip) {
  std::vector<Pt> output = subject;
  for (std::size_t edge = 0; edge < clip.size(); ++edge) {
    if (output.empty()) break;
    const Pt edge_a = clip[edge];
    const Pt edge_b = clip[(edge + 1) % clip.size()];
    const std::vector<Pt> input = output;
    output.clear();
    for (std::size_t index = 0; index < input.size(); ++index) {
      const Pt start = input[index];
      const Pt end = input[(index + 1) % input.size()];
      const bool start_in = cross(sub(edge_b, edge_a), sub(start, edge_a)) >= -1e-9;
      const bool end_in = cross(sub(edge_b, edge_a), sub(end, edge_a)) >= -1e-9;
      if (start_in && end_in) {
        output.push_back(end);
      } else if (start_in && !end_in) {
        output.push_back(line_hit(start, end, edge_a, edge_b));
      } else if (!start_in && end_in) {
        output.push_back(line_hit(start, end, edge_a, edge_b));
        output.push_back(end);
      }
    }
  }
  std::vector<Pt> cleaned;
  for (const Pt& point : output) {
    if (!cleaned.empty()) {
      const double dx = point.x - cleaned.back().x;
      const double dy = point.y - cleaned.back().y;
      if (dx * dx + dy * dy <= 1e-18) continue;
    }
    cleaned.push_back(point);
  }
  if (cleaned.size() >= 2) {
    const double dx = cleaned.front().x - cleaned.back().x;
    const double dy = cleaned.front().y - cleaned.back().y;
    if (dx * dx + dy * dy <= 1e-18) cleaned.pop_back();
  }
  return cleaned;
}

double polygon_area(const std::vector<Pt>& points) { return std::fabs(signed_area(points)); }

bool corners_inside(const OrientedBox& box, int width, int height) {
  const std::array<Pt, 4> points = corners_of(box);
  for (const Pt& point : points) {
    if (point.x < -kInsideSlack || point.y < -kInsideSlack || point.x > static_cast<double>(width) + kInsideSlack ||
        point.y > static_cast<double>(height) + kInsideSlack) {
      return false;
    }
  }
  return true;
}

OrientedBox fit_minimum_area(const std::array<Pt, 4>& points) {
  OrientedBox best;
  double best_area = 0;
  bool found = false;
  for (int edge = 0; edge < 4; ++edge) {
    const Pt delta = sub(points[static_cast<std::size_t>((edge + 1) % 4)], points[static_cast<std::size_t>(edge)]);
    const double length = std::hypot(delta.x, delta.y);
    if (!(length > 1e-12)) continue;
    const Pt axis{delta.x / length, delta.y / length};
    const Pt perp{-axis.y, axis.x};
    double min_u = 0;
    double max_u = 0;
    double min_v = 0;
    double max_v = 0;
    for (int index = 0; index < 4; ++index) {
      const double along = points[static_cast<std::size_t>(index)].x * axis.x +
                           points[static_cast<std::size_t>(index)].y * axis.y;
      const double across = points[static_cast<std::size_t>(index)].x * perp.x +
                            points[static_cast<std::size_t>(index)].y * perp.y;
      if (index == 0 || along < min_u) min_u = along;
      if (index == 0 || along > max_u) max_u = along;
      if (index == 0 || across < min_v) min_v = across;
      if (index == 0 || across > max_v) max_v = across;
    }
    double width = max_u - min_u;
    double height = max_v - min_v;
    if (!(width > 0) || !(height > 0)) continue;
    const double center_u = 0.5 * (min_u + max_u);
    const double center_v = 0.5 * (min_v + max_v);
    double angle = std::atan2(axis.y, axis.x);
    while (angle >= 3.0 * kPi / 4.0) {
      angle -= kPi / 2.0;
      std::swap(width, height);
    }
    while (angle < -kPi / 4.0) {
      angle += kPi / 2.0;
      std::swap(width, height);
    }
    const double area = width * height;
    const bool better = !found || area < best_area - 1e-9 ||
                        (std::fabs(area - best_area) <= 1e-9 && std::fabs(angle) < std::fabs(best.angle) - 1e-12) ||
                        (std::fabs(area - best_area) <= 1e-9 &&
                         std::fabs(std::fabs(angle) - std::fabs(best.angle)) <= 1e-12 && angle < best.angle);
    if (better) {
      best.cx = static_cast<float>(center_u * axis.x + center_v * perp.x);
      best.cy = static_cast<float>(center_u * axis.y + center_v * perp.y);
      best.width = static_cast<float>(width);
      best.height = static_cast<float>(height);
      best.angle = static_cast<float>(angle);
      best_area = area;
      found = true;
    }
  }
  if (!found || !std::isfinite(best.cx) || !(best.width > 0.f) || !(best.height > 0.f)) {
    throw VieError(ErrorCode::InvalidArgument, "rotated rectangle fit did not produce a positive box");
  }
  return best;
}

}  // namespace

void validate_obb_decode_config(const ObbDecodeConfig& config) {
  if (config.output_name.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "obb output_name is empty");
  }
  if (config.channel_axis != 1 && config.channel_axis != 2) {
    throw VieError(ErrorCode::InvalidArgument, "obb channel_axis must be 1 or 2");
  }
  if (config.num_classes < 1) {
    throw VieError(ErrorCode::InvalidArgument, "obb num_classes must be >= 1");
  }
  if (!config.class_selection.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "obb class_selection is unresolved");
  }
  if (!config.confidence_threshold.has_value() || !std::isfinite(*config.confidence_threshold) ||
      *config.confidence_threshold < 0.f || *config.confidence_threshold > 1.f) {
    throw VieError(ErrorCode::UnresolvedField, "obb confidence_threshold is unresolved or outside [0, 1]");
  }
  if (!config.nms_enabled.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "obb nms_enabled is unresolved");
  }
  if (!config.max_detections.has_value() || *config.max_detections < 1) {
    throw VieError(ErrorCode::UnresolvedField, "obb max_detections is unresolved");
  }
  if (!config.pre_nms_top_k.has_value() || *config.pre_nms_top_k < 1 || *config.pre_nms_top_k > 4096) {
    throw VieError(ErrorCode::UnresolvedField, "obb pre_nms_top_k is unresolved or outside 1..4096");
  }
  if (*config.nms_enabled) {
    if (!config.class_aware.has_value()) {
      throw VieError(ErrorCode::UnresolvedField, "obb class_aware is unresolved");
    }
    if (!config.iou_threshold.has_value() || !std::isfinite(*config.iou_threshold) || *config.iou_threshold < 0.f ||
        *config.iou_threshold > 1.f) {
      throw VieError(ErrorCode::UnresolvedField, "obb iou_threshold is unresolved or outside [0, 1]");
    }
  }
}

float rotated_intersection_over_union(const OrientedBox& left, const OrientedBox& right) {
  require_box(left, "rotated box");
  require_box(right, "rotated box");
  const std::vector<Pt> left_points = ccw_corners(left);
  const std::vector<Pt> right_points = ccw_corners(right);
  const double left_area = polygon_area(left_points);
  const double right_area = polygon_area(right_points);
  if (!(left_area > 0) || !(right_area > 0) || !std::isfinite(left_area) || !std::isfinite(right_area)) {
    throw VieError(ErrorCode::InvalidArgument, "rotated box area is not positive");
  }
  double intersection = polygon_area(clip_convex(left_points, right_points));
  if (!std::isfinite(intersection) || intersection < 0) {
    throw VieError(ErrorCode::InvalidArgument, "rotated intersection area is not finite");
  }
  if (intersection > left_area) intersection = left_area;
  if (intersection > right_area) intersection = right_area;
  const double union_area = left_area + right_area - intersection;
  if (!(union_area > 0) || !std::isfinite(union_area)) {
    throw VieError(ErrorCode::InvalidArgument, "rotated union area is not positive");
  }
  const double ratio = intersection / union_area;
  if (ratio <= 0) return 0.f;
  if (ratio >= 1) return 1.f;
  return static_cast<float>(ratio);
}

std::vector<int> rotated_nms(const std::vector<ObbCandidate>& sorted, float iou_threshold, bool class_aware,
                             int max_detections) {
  if (!std::isfinite(iou_threshold) || iou_threshold < 0.f || iou_threshold > 1.f) {
    throw VieError(ErrorCode::InvalidArgument, "rotated NMS IoU threshold is outside [0, 1]");
  }
  if (max_detections < 1) {
    throw VieError(ErrorCode::InvalidArgument, "rotated NMS max_detections must be >= 1");
  }
  std::vector<char> suppressed(sorted.size(), 0);
  std::vector<int> kept;
  kept.reserve(static_cast<std::size_t>(std::min(max_detections, static_cast<int>(sorted.size()))));
  for (std::size_t index = 0; index < sorted.size(); ++index) {
    if (suppressed[index] != 0) continue;
    if (!std::isfinite(sorted[index].score)) {
      throw VieError(ErrorCode::InvalidArgument, "rotated NMS score is not finite");
    }
    kept.push_back(static_cast<int>(index));
    if (static_cast<int>(kept.size()) == max_detections) break;
    for (std::size_t other = index + 1; other < sorted.size(); ++other) {
      if (suppressed[other] != 0) continue;
      if (class_aware && sorted[other].class_id != sorted[index].class_id) continue;
      if (rotated_intersection_over_union(sorted[index].box, sorted[other].box) >= iou_threshold) {
        suppressed[other] = 1;
      }
    }
  }
  return kept;
}

bool map_obb_to_original(const OrientedBox& model_box, const GeometryTransform& geometry, OrientedBox& original) {
  require_box(model_box, "model obb");
  if (geometry.original_width <= 0 || geometry.original_height <= 0 || geometry.scale_x == 0.0 ||
      geometry.scale_y == 0.0 || !std::isfinite(geometry.scale_x) || !std::isfinite(geometry.scale_y)) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale or original size is invalid");
  }
  const double scale_x = geometry.scale_x;
  const double scale_y = geometry.scale_y;
  const double origin_x = static_cast<double>(geometry_model_origin_x(geometry));
  const double origin_y = static_cast<double>(geometry_model_origin_y(geometry));
  const auto map_point = [&](Pt point) {
    return Pt{(point.x - origin_x) / scale_x, (point.y - origin_y) / scale_y};
  };
  const std::array<Pt, 4> model_points = corners_of(model_box);
  std::array<Pt, 4> mapped{};
  for (int index = 0; index < 4; ++index) mapped[static_cast<std::size_t>(index)] = map_point(model_points[static_cast<std::size_t>(index)]);
  for (const Pt& point : mapped) {
    if (point.x < -kInsideSlack || point.y < -kInsideSlack ||
        point.x > static_cast<double>(geometry.original_width) + kInsideSlack ||
        point.y > static_cast<double>(geometry.original_height) + kInsideSlack) {
      return false;
    }
  }
  const double relative = std::fabs(scale_x - scale_y) / std::max(std::fabs(scale_x), std::fabs(scale_y));
  if (relative <= kIsotropicRelative) {
    const double scale = 0.5 * (scale_x + scale_y);
    original.cx = static_cast<float>((static_cast<double>(model_box.cx) - origin_x) / scale_x);
    original.cy = static_cast<float>((static_cast<double>(model_box.cy) - origin_y) / scale_y);
    original.width = static_cast<float>(static_cast<double>(model_box.width) / scale);
    original.height = static_cast<float>(static_cast<double>(model_box.height) / scale);
    original.angle = model_box.angle;
  } else {
    original = fit_minimum_area(mapped);
  }
  require_box(original, "mapped obb");
  return corners_inside(original, geometry.original_width, geometry.original_height);
}

}  // namespace vie
