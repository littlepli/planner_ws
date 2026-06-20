// mapping_demo.cpp
//
// Task 15 environment upgrade: limited-FOV sensor + incremental occupancy / ESDF
// mapping, so the "perception" is no longer cheating.
//
// The robot does NOT know the obstacles in advance.  It carries a planar
// laser-like sensor with limited range and a limited field of view.  As it moves
// it casts a fan of rays (with occlusion), and writes what it sees into an
// incremental occupancy grid (UNKNOWN / FREE / OCCUPIED).  An ESDF is built
// incrementally from the KNOWN occupied cells with a chamfer distance transform.
//
//   perception : ray-cast sensor -> incremental occupancy grid (+ ESDF)
//   planning   : Kinodynamic A* + EGO B-spline plan ONLY on the known map
//                (unknown = optimistically free); a newly discovered obstacle that
//                blocks the look-ahead triggers a replan around it.
//   control    : PD + feed-forward tracking of the planned trajectory.
//
// Everything runs in the z = z0 plane to keep the occupancy grid and the sensor
// footprint easy to read in RViz (a 2D cost-map style local planner).  Collisions
// are also checked against the TRUE world so the reported safety is honest.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

using namespace std::chrono_literals;

struct Vec3 { double x; double y; double z; };
struct BoxObstacle { double x; double y; double z; double sx; double sy; double sz; };
struct KinoNode { Vec3 pos; Vec3 vel; double time; double g_score; double f_score; int parent; long long key; };
struct QueueItem { double f_score; int node_id; bool operator<(const QueueItem & o) const { return f_score > o.f_score; } };
struct DesiredState { Vec3 pos; Vec3 vel; Vec3 acc; };

enum CellState : int8_t { UNKNOWN = -1, FREE = 0, OCC = 1 };

class MappingDemoNode : public rclcpp::Node
{
public:
  MappingDemoNode() : Node("mapping_demo_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/mapping_markers", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/mapping_demo/odom", 10);

    initParams();
    generateTrueObstacles();      // the hidden ground-truth world
    initGrid();

    robot_pos_ = start_pos_;
    robot_vel_ = Vec3{0.0, 0.0, 0.0};
    sensorScan();                 // first observation from the start pose
    rebuildEsdf();
    if (planFromCurrent(true)) {
      planning_ok_ = true;
      RCLCPP_INFO(this->get_logger(), "Initial plan on known map ready. Starting loop.");
    } else {
      RCLCPP_WARN(this->get_logger(), "Initial plan failed (will retry while mapping).");
    }

    timer_ = this->create_wall_timer(20ms, std::bind(&MappingDemoNode::timerCallback, this));
    RCLCPP_INFO(this->get_logger(), "mapping_demo_node started (limited-FOV sensor + incremental map).");
  }

private:
  // ------------------------------------------------------------------ params
  void initParams()
  {
    // Expanded arena: 16 x 14 m (2D plane).
    x_min_ = -8.0; x_max_ = 8.0; y_min_ = -7.0; y_max_ = 7.0;
    z0_ = 1.0;
    map_res_ = 0.20;

    max_vel_ = 2.5; max_acc_ = 2.0; dt_ = 0.4;
    pos_resolution_ = 0.4; vel_resolution_ = 0.5;
    nx_ = static_cast<int>((x_max_ - x_min_) / pos_resolution_);
    ny_ = static_cast<int>((y_max_ - y_min_) / pos_resolution_);
    nv_ = static_cast<int>(2.0 * max_vel_ / vel_resolution_) + 1;
    primitive_check_num_ = 6; goal_tolerance_ = 0.5; safety_margin_ = 0.22;
    max_search_time_ = 14.0; max_expand_num_ = 40000;

    start_pos_ = Vec3{-7.0, -6.0, z0_}; start_vel_ = Vec3{0.0, 0.0, 0.0};
    goal_pos_ = Vec3{7.0, 6.0, z0_};

    // EGO back-end (collision gradient from the known-map ESDF, in-plane only).
    opt_iterations_ = 250; replan_iterations_ = 120; opt_step_ = 0.06;
    lambda_smooth_ = 1.0; lambda_collision_ = 2.2; d_safe_ = 0.55;
    bspline_samples_per_segment_ = 12; cruise_speed_ = 1.3;

    // sensor: limited range + limited field of view, with occlusion.
    sensor_range_ = this->declare_parameter<double>("sensor_range", 3.5);
    sensor_fov_deg_ = this->declare_parameter<double>("sensor_fov_deg", 120.0);
    ray_step_ = map_res_ * 0.5;
    look_ahead_ = 1.6;            // seconds of trajectory checked for newly-seen blockage

    // controller
    ctrl_dt_ = 0.02; elapsed_time_ = 0.0; total_time_ = 0.0;
    kp_ = 4.0; kd_ = 3.6; ctrl_max_acc_ = 6.0; ctrl_max_vel_ = 3.0;
    reached_goal_ = false; planning_ok_ = false; step_count_ = 0;
    replan_period_ = 0.2; last_replan_time_ = 0.0; wall_clock_ = 0.0;
    replan_count_ = 0; full_replan_count_ = 0;
    err_sum_sq_ = 0.0; err_max_ = 0.0; err_count_ = 0;
    exec_collision_free_ = true; exec_collision_steps_ = 0;
  }

  void generateTrueObstacles()
  {
    true_obstacles_.clear();
    std::mt19937 rng(12);
    std::uniform_real_distribution<double> pos_x(-7.2, 7.2);
    std::uniform_real_distribution<double> pos_y(-6.2, 6.2);
    std::uniform_real_distribution<double> size_xy(0.5, 0.95);
    for (int i = 0; i < 40; ++i) {
      BoxObstacle obs;
      obs.x = pos_x(rng); obs.y = pos_y(rng);
      obs.sx = size_xy(rng); obs.sy = size_xy(rng);
      obs.sz = 2.0; obs.z = 1.0;
      if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
      if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
      true_obstacles_.push_back(obs);
    }
    true_obstacles_.push_back(BoxObstacle{-1.2, -0.5, 1.0, 1.0, 2.8, 2.0});
    true_obstacles_.push_back(BoxObstacle{ 1.5,  1.2, 1.0, 2.4, 1.0, 2.0});
    true_obstacles_.push_back(BoxObstacle{ 2.5, -1.8, 1.0, 1.0, 2.6, 2.0});
  }

  void initGrid()
  {
    gnx_ = static_cast<int>((x_max_ - x_min_) / map_res_) + 1;
    gny_ = static_cast<int>((y_max_ - y_min_) / map_res_) + 1;
    occ_.assign(static_cast<size_t>(gnx_) * gny_, UNKNOWN);
    esdf_.assign(static_cast<size_t>(gnx_) * gny_, 1e3);
  }

  // ------------------------------------------------------------------ helpers
  double distance2D(double x1, double y1, double x2, double y2) const
  { const double dx = x1 - x2, dy = y1 - y2; return std::sqrt(dx * dx + dy * dy); }
  double norm(const Vec3 & v) const { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
  double dist(const Vec3 & a, const Vec3 & b) const
  { const double dx = a.x - b.x, dy = a.y - b.y; return std::sqrt(dx * dx + dy * dy); }
  bool inArena(double x, double y) const
  { return x >= x_min_ && x < x_max_ && y >= y_min_ && y < y_max_; }
  int cellIndex(int ix, int iy) const { return iy * gnx_ + ix; }
  bool worldToCell(double x, double y, int & ix, int & iy) const
  {
    ix = static_cast<int>((x - x_min_) / map_res_);
    iy = static_cast<int>((y - y_min_) / map_res_);
    return ix >= 0 && ix < gnx_ && iy >= 0 && iy < gny_;
  }
  // Ground-truth occupancy in the plane (used ONLY by the sensor model).
  bool trueOccupied(double x, double y) const
  {
    for (const auto & o : true_obstacles_)
      if (std::abs(x - o.x) <= o.sx / 2.0 && std::abs(y - o.y) <= o.sy / 2.0) return true;
    return false;
  }

  // ============================ PERCEPTION ============================ //
  // Cast a fan of rays from the robot within [heading +/- fov/2], up to sensor_range_.
  // Each ray marks FREE cells until it hits a true obstacle (-> OCC, then stops:
  // everything behind stays UNKNOWN -> occlusion shadow).
  void sensorScan()
  {
    double heading;
    if (norm(robot_vel_) > 0.3) heading = std::atan2(robot_vel_.y, robot_vel_.x);
    else heading = std::atan2(goal_pos_.y - robot_pos_.y, goal_pos_.x - robot_pos_.x);
    last_heading_ = heading;

    const double fov = sensor_fov_deg_ * M_PI / 180.0;
    const int num_rays = static_cast<int>(sensor_fov_deg_ / 1.5) + 1;
    for (int r = 0; r < num_rays; ++r) {
      const double a = -fov / 2.0 + fov * r / std::max(1, num_rays - 1);
      const double ang = heading + a;
      const double cx = std::cos(ang), sy = std::sin(ang);
      for (double rr = map_res_; rr <= sensor_range_; rr += ray_step_) {
        const double px = robot_pos_.x + cx * rr;
        const double py = robot_pos_.y + sy * rr;
        int ix, iy;
        if (!worldToCell(px, py, ix, iy)) break;
        const int idx = cellIndex(ix, iy);
        if (trueOccupied(px, py)) { occ_[idx] = OCC; break; }   // hit -> stop ray
        if (occ_[idx] != OCC) occ_[idx] = FREE;
      }
    }
  }

  // Incremental ESDF on the known map: distance (m) to the nearest OCCUPIED cell,
  // via a two-pass chamfer distance transform.  Unknown/free cells are non-sources,
  // so unmapped space is treated as obstacle-free (optimistic planning).
  void rebuildEsdf()
  {
    const double BIG = 1e3;
    std::vector<double> d(occ_.size(), BIG);
    for (size_t i = 0; i < occ_.size(); ++i) if (occ_[i] == OCC) d[i] = 0.0;
    const double c1 = 1.0, c2 = std::sqrt(2.0);
    auto at = [&](int ix, int iy) -> double & { return d[cellIndex(ix, iy)]; };
    for (int iy = 0; iy < gny_; ++iy)
      for (int ix = 0; ix < gnx_; ++ix) {
        double v = at(ix, iy);
        if (ix > 0) v = std::min(v, at(ix - 1, iy) + c1);
        if (iy > 0) v = std::min(v, at(ix, iy - 1) + c1);
        if (ix > 0 && iy > 0) v = std::min(v, at(ix - 1, iy - 1) + c2);
        if (ix < gnx_ - 1 && iy > 0) v = std::min(v, at(ix + 1, iy - 1) + c2);
        at(ix, iy) = v;
      }
    for (int iy = gny_ - 1; iy >= 0; --iy)
      for (int ix = gnx_ - 1; ix >= 0; --ix) {
        double v = at(ix, iy);
        if (ix < gnx_ - 1) v = std::min(v, at(ix + 1, iy) + c1);
        if (iy < gny_ - 1) v = std::min(v, at(ix, iy + 1) + c1);
        if (ix < gnx_ - 1 && iy < gny_ - 1) v = std::min(v, at(ix + 1, iy + 1) + c2);
        if (ix > 0 && iy < gny_ - 1) v = std::min(v, at(ix - 1, iy + 1) + c2);
        at(ix, iy) = v;
      }
    for (size_t i = 0; i < d.size(); ++i) esdf_[i] = d[i] * map_res_;   // cells -> metres
  }
  // Bilinear interpolation of the known-map ESDF.
  double knownClearance(double x, double y) const
  {
    const double fx = (x - x_min_) / map_res_, fy = (y - y_min_) / map_res_;
    int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, gnx_ - 2);
    int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, gny_ - 2);
    const double tx = std::clamp(fx - ix, 0.0, 1.0), ty = std::clamp(fy - iy, 0.0, 1.0);
    const double c00 = esdf_[cellIndex(ix, iy)], c10 = esdf_[cellIndex(ix + 1, iy)];
    const double c01 = esdf_[cellIndex(ix, iy + 1)], c11 = esdf_[cellIndex(ix + 1, iy + 1)];
    return (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty;
  }
  void knownClearanceGrad(double x, double y, double & gx, double & gy) const
  {
    const double h = map_res_;
    gx = (knownClearance(x + h, y) - knownClearance(x - h, y)) / (2 * h);
    gy = (knownClearance(x, y + h) - knownClearance(x, y - h)) / (2 * h);
  }
  bool knownCollide(const Vec3 & p) const
  { return !inArena(p.x, p.y) || knownClearance(p.x, p.y) < safety_margin_; }
  bool trueCollide(const Vec3 & p) const
  {
    if (!inArena(p.x, p.y)) return true;
    for (const auto & o : true_obstacles_)
      if (std::abs(p.x - o.x) <= o.sx / 2.0 + safety_margin_ &&
          std::abs(p.y - o.y) <= o.sy / 2.0 + safety_margin_) return true;
    return false;
  }

  // ============================ PLANNING ============================== //
  bool planFromCurrent(bool full)
  {
    if (full) {
      start_pos_ = robot_pos_; start_vel_ = robot_vel_;
      if (!runKinoAStar()) return false;
      generateControlPoints();
      current_cps_ = optimizeEgoCps(cps_, opt_iterations_);
      full_replan_count_++;
    } else {
      if (current_cps_.size() < 4) return false;
      current_cps_ = optimizeEgoCps(current_cps_, replan_iterations_);
    }
    opt_path_ = sampleBspline(current_cps_);
    if (opt_path_.size() < 2) return false;
    timeAllocate();
    reAnchorTime();
    replan_count_++;
    return true;
  }
  void reAnchorTime()
  {
    double best = std::numeric_limits<double>::infinity(); size_t bi = 0;
    for (size_t i = 0; i < ref_pos_.size(); ++i) {
      const double dd = dist(robot_pos_, toVec(ref_pos_[i]));
      if (dd < best) { best = dd; bi = i; }
    }
    if (!ref_time_.empty()) elapsed_time_ = ref_time_[bi];
  }
  // Is the upcoming reference (next look_ahead_ seconds) blocked by the known map?
  bool lookAheadBlocked() const
  {
    for (double t = elapsed_time_; t <= elapsed_time_ + look_ahead_; t += 0.1) {
      DesiredState d = referenceAt(t);
      if (knownCollide(d.pos)) return true;
    }
    return false;
  }

  // ----- front-end Kinodynamic A* (in-plane, queries the KNOWN map) -----
  bool isStateValid(const Vec3 & p, const Vec3 & v) const
  { return inArena(p.x, p.y) && !knownCollide(p) && norm(v) <= max_vel_ + 1e-6; }
  long long stateKey(const Vec3 & p, const Vec3 & v) const
  {
    int ix = static_cast<int>((p.x - x_min_) / pos_resolution_);
    int iy = static_cast<int>((p.y - y_min_) / pos_resolution_);
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_) return -1;
    int ivx = static_cast<int>(std::round((v.x + max_vel_) / vel_resolution_));
    int ivy = static_cast<int>(std::round((v.y + max_vel_) / vel_resolution_));
    if (ivx < 0 || ivx >= nv_ || ivy < 0 || ivy >= nv_) return -1;
    long long key = ix; key = key * ny_ + iy; key = key * nv_ + ivx; key = key * nv_ + ivy;
    return key;
  }
  Vec3 propagatePos(const Vec3 & p, const Vec3 & v, const Vec3 & a, double t) const
  { return Vec3{p.x + v.x * t + 0.5 * a.x * t * t, p.y + v.y * t + 0.5 * a.y * t * t, z0_}; }
  Vec3 propagateVel(const Vec3 & v, const Vec3 & a, double t) const
  { return Vec3{v.x + a.x * t, v.y + a.y * t, 0.0}; }
  bool checkPrimitiveCollision(const Vec3 & p, const Vec3 & v, const Vec3 & a) const
  {
    for (int i = 1; i <= primitive_check_num_; ++i) {
      const double t = dt_ * static_cast<double>(i) / primitive_check_num_;
      if (!isStateValid(propagatePos(p, v, a, t), propagateVel(v, a, t))) return false;
    }
    return true;
  }
  std::vector<Vec3> generateControlSamples() const
  {
    std::vector<Vec3> c;
    const std::array<double, 3> vals = {-max_acc_, 0.0, max_acc_};
    for (double ax : vals) for (double ay : vals) {
      Vec3 a{ax, ay, 0.0}; if (norm(a) <= max_acc_ * 1.5) c.push_back(a);
    }
    return c;
  }
  double heuristic(const Vec3 & p, const Vec3 & v) const
  {
    const double d = dist(p, goal_pos_);
    Vec3 tg{goal_pos_.x - p.x, goal_pos_.y - p.y, 0.0};
    const double dn = std::max(d, 1e-3);
    Vec3 dv{max_vel_ * tg.x / dn, max_vel_ * tg.y / dn, 0.0};
    return d / max_vel_ + 0.08 * dist(v, dv);
  }
  bool runKinoAStar()
  {
    if (!isStateValid(start_pos_, start_vel_)) {
      // If we are momentarily inside the safety margin of a just-seen obstacle,
      // still try to plan by relaxing the start check (controller will recover).
      if (!inArena(start_pos_.x, start_pos_.y)) return false;
    }
    const auto controls = generateControlSamples();
    std::vector<KinoNode> nodes;
    std::priority_queue<QueueItem> open;
    std::unordered_map<long long, double> best_g;
    std::unordered_map<long long, int> best_id;
    std::unordered_set<long long> closed;
    long long sk = stateKey(start_pos_, start_vel_);
    KinoNode s; s.pos = start_pos_; s.vel = start_vel_; s.time = 0.0;
    s.g_score = 0.0; s.f_score = heuristic(start_pos_, start_vel_); s.parent = -1; s.key = sk;
    nodes.push_back(s); best_g[sk] = 0.0; best_id[sk] = 0; open.push(QueueItem{s.f_score, 0});
    int final_id = -1, expand = 0;
    while (!open.empty() && expand < max_expand_num_) {
      QueueItem it = open.top(); open.pop();
      const KinoNode cur = nodes[it.node_id];
      auto bit = best_id.find(cur.key);
      if (bit == best_id.end() || bit->second != it.node_id) continue;
      if (closed.count(cur.key)) continue;
      closed.insert(cur.key); expand++;
      if (dist(cur.pos, goal_pos_) < goal_tolerance_) { final_id = it.node_id; break; }
      if (cur.time > max_search_time_) continue;
      for (const auto & acc : controls) {
        if (!checkPrimitiveCollision(cur.pos, cur.vel, acc)) continue;
        Vec3 np = propagatePos(cur.pos, cur.vel, acc, dt_);
        Vec3 nv = propagateVel(cur.vel, acc, dt_);
        long long nk = stateKey(np, nv);
        if (nk < 0 || closed.count(nk)) continue;
        const double ng = cur.g_score + dt_ + 0.03 * norm(acc) * norm(acc) * dt_;
        auto git = best_g.find(nk);
        if (git == best_g.end() || ng < git->second) {
          KinoNode n; n.pos = np; n.vel = nv; n.time = cur.time + dt_;
          n.g_score = ng; n.f_score = ng + heuristic(np, nv); n.parent = it.node_id; n.key = nk;
          int id = static_cast<int>(nodes.size()); nodes.push_back(n);
          best_g[nk] = ng; best_id[nk] = id; open.push(QueueItem{n.f_score, id});
        }
      }
    }
    kino_path_.clear();
    if (final_id < 0) return false;
    std::vector<int> ids; int id = final_id;
    while (id >= 0) { ids.push_back(id); id = nodes[id].parent; }
    std::reverse(ids.begin(), ids.end());
    for (int nid : ids) {
      geometry_msgs::msg::Point p; p.x = nodes[nid].pos.x; p.y = nodes[nid].pos.y; p.z = z0_;
      kino_path_.push_back(p);
    }
    return true;
  }

  // ----- back-end EGO (collision gradient from the known-map ESDF) -----
  void generateControlPoints()
  {
    cps_.clear();
    if (kino_path_.size() < 2) return;
    geometry_msgs::msg::Point goal_pt = toPoint(goal_pos_);
    cps_.push_back(kino_path_.front()); cps_.push_back(kino_path_.front());
    for (const auto & p : kino_path_) cps_.push_back(p);
    cps_.push_back(goal_pt); cps_.push_back(goal_pt);
  }
  geometry_msgs::msg::Point cubicBsplinePoint(
    const geometry_msgs::msg::Point & p0, const geometry_msgs::msg::Point & p1,
    const geometry_msgs::msg::Point & p2, const geometry_msgs::msg::Point & p3, double u) const
  {
    const double u2 = u * u, u3 = u2 * u;
    const double b0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0;
    const double b1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0;
    const double b2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0;
    const double b3 = u3 / 6.0;
    geometry_msgs::msg::Point p;
    p.x = b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x;
    p.y = b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y;
    p.z = z0_;
    return p;
  }
  std::vector<geometry_msgs::msg::Point> sampleBspline(
    const std::vector<geometry_msgs::msg::Point> & cps) const
  {
    std::vector<geometry_msgs::msg::Point> path;
    if (cps.size() < 4) return path;
    for (size_t i = 0; i + 3 < cps.size(); ++i)
      for (int j = 0; j < bspline_samples_per_segment_; ++j) {
        const double u = static_cast<double>(j) / bspline_samples_per_segment_;
        path.push_back(cubicBsplinePoint(cps[i], cps[i + 1], cps[i + 2], cps[i + 3], u));
      }
    path.push_back(cps.back());
    return path;
  }
  std::vector<geometry_msgs::msg::Point> optimizeEgoCps(
    const std::vector<geometry_msgs::msg::Point> & seed, int iters)
  {
    auto q = seed;
    for (int iter = 0; iter < iters; ++iter) {
      std::vector<Vec3> grad(q.size(), Vec3{0.0, 0.0, 0.0});
      for (size_t i = 1; i + 1 < q.size(); ++i) {
        Vec3 a{q[i + 1].x - 2.0 * q[i].x + q[i - 1].x, q[i + 1].y - 2.0 * q[i].y + q[i - 1].y, 0.0};
        grad[i - 1].x += lambda_smooth_ * 2.0 * a.x; grad[i - 1].y += lambda_smooth_ * 2.0 * a.y;
        grad[i].x     -= lambda_smooth_ * 4.0 * a.x; grad[i].y     -= lambda_smooth_ * 4.0 * a.y;
        grad[i + 1].x += lambda_smooth_ * 2.0 * a.x; grad[i + 1].y += lambda_smooth_ * 2.0 * a.y;
      }
      for (size_t i = 0; i < q.size(); ++i) {
        const double d = knownClearance(q[i].x, q[i].y);
        if (d >= d_safe_) continue;
        double gx, gy; knownClearanceGrad(q[i].x, q[i].y, gx, gy);
        const double coef = -2.0 * lambda_collision_ * (d_safe_ - d);
        grad[i].x += coef * gx; grad[i].y += coef * gy;
      }
      for (size_t i = 2; i + 2 < q.size(); ++i) {
        q[i].x -= opt_step_ * grad[i].x; q[i].y -= opt_step_ * grad[i].y;
        q[i].x = std::clamp(q[i].x, x_min_ + 0.2, x_max_ - 0.2);
        q[i].y = std::clamp(q[i].y, y_min_ + 0.2, y_max_ - 0.2);
        q[i].z = z0_;
      }
    }
    return q;
  }
  void timeAllocate()
  {
    ref_pos_ = opt_path_;
    ref_time_.assign(ref_pos_.size(), 0.0);
    for (size_t i = 1; i < ref_pos_.size(); ++i)
      ref_time_[i] = ref_time_[i - 1] + dist(toVec(ref_pos_[i]), toVec(ref_pos_[i - 1])) / cruise_speed_;
    total_time_ = ref_time_.empty() ? 0.0 : ref_time_.back();
  }
  DesiredState referenceAt(double t) const
  {
    DesiredState d{}; const size_t n = ref_pos_.size();
    if (n == 0) { d.pos = robot_pos_; return d; }
    if (t <= 0.0) { d.pos = toVec(ref_pos_.front()); return d; }
    if (t >= total_time_) { d.pos = toVec(ref_pos_.back()); return d; }
    size_t i = 1; while (i < n && ref_time_[i] < t) ++i; if (i >= n) i = n - 1;
    const double t0 = ref_time_[i - 1], t1 = ref_time_[i];
    const double sdt = std::max(t1 - t0, 1e-6), a = (t - t0) / sdt;
    d.pos = Vec3{ref_pos_[i - 1].x + a * (ref_pos_[i].x - ref_pos_[i - 1].x),
                 ref_pos_[i - 1].y + a * (ref_pos_[i].y - ref_pos_[i - 1].y), z0_};
    d.vel = Vec3{(ref_pos_[i].x - ref_pos_[i - 1].x) / sdt, (ref_pos_[i].y - ref_pos_[i - 1].y) / sdt, 0.0};
    return d;
  }
  Vec3 toVec(const geometry_msgs::msg::Point & p) const { return Vec3{p.x, p.y, z0_}; }
  geometry_msgs::msg::Point toPoint(const Vec3 & v) const
  { geometry_msgs::msg::Point p; p.x = v.x; p.y = v.y; p.z = z0_; return p; }

  // ============================ CONTROL =============================== //
  Vec3 clampNorm(const Vec3 & v, double m) const
  { const double n = norm(v); if (n <= m || n < 1e-9) return v; const double s = m / n; return Vec3{v.x * s, v.y * s, 0.0}; }
  void controlStep()
  {
    DesiredState d = referenceAt(elapsed_time_);
    Vec3 pe{d.pos.x - robot_pos_.x, d.pos.y - robot_pos_.y, 0.0};
    Vec3 ve{d.vel.x - robot_vel_.x, d.vel.y - robot_vel_.y, 0.0};
    Vec3 acc{kp_ * pe.x + kd_ * ve.x, kp_ * pe.y + kd_ * ve.y, 0.0};
    acc = clampNorm(acc, ctrl_max_acc_);
    robot_pos_.x += robot_vel_.x * ctrl_dt_ + 0.5 * acc.x * ctrl_dt_ * ctrl_dt_;
    robot_pos_.y += robot_vel_.y * ctrl_dt_ + 0.5 * acc.y * ctrl_dt_ * ctrl_dt_;
    robot_vel_.x += acc.x * ctrl_dt_; robot_vel_.y += acc.y * ctrl_dt_;
    robot_vel_ = clampNorm(robot_vel_, ctrl_max_vel_);

    current_desired_ = d;
    current_error_ = dist(robot_pos_, d.pos);
    err_sum_sq_ += current_error_ * current_error_; err_count_++;
    err_max_ = std::max(err_max_, current_error_);
    if (trueCollide(robot_pos_)) { exec_collision_free_ = false; exec_collision_steps_++; }

    geometry_msgs::msg::Point ap = toPoint(robot_pos_); actual_path_.push_back(ap);
    if (actual_path_.size() > 4000) actual_path_.erase(actual_path_.begin());

    if (!reached_goal_ && dist(robot_pos_, goal_pos_) < 0.3 && norm(robot_vel_) < 0.5) {
      reached_goal_ = true;
      const double rms = (err_count_ > 0) ? std::sqrt(err_sum_sq_ / err_count_) : 0.0;
      const int disc = discoveredObstacleCount();
      RCLCPP_INFO(this->get_logger(), "============ MAPPING CLOSED-LOOP REPORT ============");
      RCLCPP_INFO(this->get_logger(), "sensor: range %.1f m, FOV %.0f deg", sensor_range_, sensor_fov_deg_);
      RCLCPP_INFO(this->get_logger(), "goal reached at t = %.2f s", wall_clock_);
      RCLCPP_INFO(this->get_logger(), "obstacles discovered: %d / %zu (rest never entered the path)",
        disc, true_obstacles_.size());
      RCLCPP_INFO(this->get_logger(), "replans: %d (full front-end replans: %d)", replan_count_, full_replan_count_);
      RCLCPP_INFO(this->get_logger(), "RMS tracking error: %.4f m, max: %.4f m", rms, err_max_);
      RCLCPP_INFO(this->get_logger(), "executed path collision-free (vs TRUE world): %s (%d steps)",
        exec_collision_free_ ? "true" : "false", exec_collision_steps_);
      RCLCPP_INFO(this->get_logger(), "===================================================");
    }
  }
  int discoveredObstacleCount() const
  {
    int n = 0;
    for (const auto & o : true_obstacles_) {
      bool seen = false;
      for (double dx = -o.sx / 2.0; dx <= o.sx / 2.0 + 1e-6 && !seen; dx += map_res_)
        for (double dy = -o.sy / 2.0; dy <= o.sy / 2.0 + 1e-6 && !seen; dy += map_res_) {
          int ix, iy;
          if (worldToCell(o.x + dx, o.y + dy, ix, iy) && occ_[cellIndex(ix, iy)] == OCC) seen = true;
        }
      if (seen) n++;
    }
    return n;
  }

  void publishOdometry()
  {
    nav_msgs::msg::Odometry odom;
    odom.header.frame_id = "map"; odom.header.stamp = this->now(); odom.child_frame_id = "base_link";
    odom.pose.pose.position = toPoint(robot_pos_); odom.pose.pose.orientation.w = 1.0;
    odom.twist.twist.linear.x = robot_vel_.x; odom.twist.twist.linear.y = robot_vel_.y;
    odom_pub_->publish(odom);
  }

  // ============================ MAIN LOOP ============================= //
  void timerCallback()
  {
    if (planning_ok_ && !reached_goal_) {
      sensorScan();                                    // perception
      if (wall_clock_ - last_replan_time_ >= replan_period_) {
        rebuildEsdf();                                 // incremental ESDF
        planFromCurrent(false);                        // fast EGO local re-opt
        if (lookAheadBlocked()) planFromCurrent(true); // discovered blockage -> full replan
        last_replan_time_ = wall_clock_;
      }
      controlStep();                                   // control
      elapsed_time_ += ctrl_dt_; wall_clock_ += ctrl_dt_;
    } else if (!planning_ok_ && !reached_goal_) {
      // keep mapping in place until an initial plan succeeds
      sensorScan(); rebuildEsdf();
      if (planFromCurrent(true)) planning_ok_ = true;
    }
    publishOdometry();
    publishMarkers();
    if (++step_count_ % 50 == 0 && planning_ok_) {
      RCLCPP_INFO(this->get_logger(),
        "t: %.2f s | pos [%.2f, %.2f] | err %.3f | discovered %d/%zu | replans %d(full %d)",
        wall_clock_, robot_pos_.x, robot_pos_.y, current_error_,
        discoveredObstacleCount(), true_obstacles_.size(), replan_count_, full_replan_count_);
    }
  }

  // ============================ MARKERS ============================== //
  void publishMarkers()
  {
    visualization_msgs::msg::MarkerArray arr;
    addTrueObstacles(arr);       // faint ghost of the real world
    addKnownMap(arr);            // solid discovered occupancy
    addSensorFan(arr);           // current field of view
    addLine(arr, "ref_path", 300, ref_pos_, 0.07, 1.0, 0.85, 0.05);
    addLine(arr, "actual_path", 301, actual_path_, 0.06, 0.0, 0.9, 1.0);
    arr.markers.push_back(sphere("sg", 400, start_pos_, 0.32, 0.0, 1.0, 0.0, 1.0));
    arr.markers.push_back(sphere("sg", 401, goal_pos_, 0.32, 0.0, 0.2, 1.0, 1.0));
    arr.markers.push_back(sphere("robot", 402, robot_pos_, 0.3, 1.0, 0.55, 0.0, 1.0));
    marker_pub_->publish(arr);
  }
  void addTrueObstacles(visualization_msgs::msg::MarkerArray & arr)
  {
    int id = 0;
    for (const auto & o : true_obstacles_) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map"; m.header.stamp = this->now();
      m.ns = "true_world"; m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE; m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = o.x; m.pose.position.y = o.y; m.pose.position.z = o.z; m.pose.orientation.w = 1.0;
      m.scale.x = o.sx; m.scale.y = o.sy; m.scale.z = o.sz;
      m.color.r = 0.5; m.color.g = 0.5; m.color.b = 0.5; m.color.a = 0.18;   // ghostly
      arr.markers.push_back(m);
    }
  }
  void addKnownMap(visualization_msgs::msg::MarkerArray & arr)
  {
    visualization_msgs::msg::Marker occ;
    occ.header.frame_id = "map"; occ.header.stamp = this->now();
    occ.ns = "known_occupied"; occ.id = 100;
    occ.type = visualization_msgs::msg::Marker::CUBE_LIST; occ.action = visualization_msgs::msg::Marker::ADD;
    occ.scale.x = map_res_; occ.scale.y = map_res_; occ.scale.z = 2.0; occ.pose.orientation.w = 1.0;
    occ.color.r = 0.85; occ.color.g = 0.1; occ.color.b = 0.1; occ.color.a = 0.9;
    visualization_msgs::msg::Marker fr;
    fr.header.frame_id = "map"; fr.header.stamp = this->now();
    fr.ns = "known_free"; fr.id = 101;
    fr.type = visualization_msgs::msg::Marker::CUBE_LIST; fr.action = visualization_msgs::msg::Marker::ADD;
    fr.scale.x = map_res_; fr.scale.y = map_res_; fr.scale.z = 0.02; fr.pose.orientation.w = 1.0;
    fr.color.r = 0.2; fr.color.g = 0.45; fr.color.b = 0.2; fr.color.a = 0.25;
    for (int iy = 0; iy < gny_; ++iy)
      for (int ix = 0; ix < gnx_; ++ix) {
        const int8_t s = occ_[cellIndex(ix, iy)];
        if (s == UNKNOWN) continue;
        geometry_msgs::msg::Point p;
        p.x = x_min_ + (ix + 0.5) * map_res_; p.y = y_min_ + (iy + 0.5) * map_res_;
        if (s == OCC) { p.z = 1.0; occ.points.push_back(p); }
        else { p.z = 0.01; fr.points.push_back(p); }
      }
    arr.markers.push_back(fr);
    arr.markers.push_back(occ);
  }
  void addSensorFan(visualization_msgs::msg::MarkerArray & arr)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = "sensor_fov"; m.id = 200;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP; m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.03; m.pose.orientation.w = 1.0;
    m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.6;
    const double fov = sensor_fov_deg_ * M_PI / 180.0;
    geometry_msgs::msg::Point c; c.x = robot_pos_.x; c.y = robot_pos_.y; c.z = z0_;
    m.points.push_back(c);
    const int seg = 24;
    for (int i = 0; i <= seg; ++i) {
      const double ang = last_heading_ - fov / 2.0 + fov * i / seg;
      geometry_msgs::msg::Point p;
      p.x = robot_pos_.x + sensor_range_ * std::cos(ang);
      p.y = robot_pos_.y + sensor_range_ * std::sin(ang); p.z = z0_;
      m.points.push_back(p);
    }
    m.points.push_back(c);
    arr.markers.push_back(m);
  }
  visualization_msgs::msg::Marker sphere(const std::string & ns, int id, const Vec3 & p,
    double s, double r, double g, double b, double a)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = ns; m.id = id; m.type = visualization_msgs::msg::Marker::SPHERE; m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position = toPoint(p); m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = s; m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
    return m;
  }
  void addLine(visualization_msgs::msg::MarkerArray & arr, const std::string & ns, int id,
    const std::vector<geometry_msgs::msg::Point> & pts, double w, double r, double g, double b)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = ns; m.id = id; m.type = visualization_msgs::msg::Marker::LINE_STRIP; m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = w; m.pose.orientation.w = 1.0; m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;
    m.points = pts;
    arr.markers.push_back(m);
  }

  // ------------------------------------------------------------------ data
  double x_min_, x_max_, y_min_, y_max_, z0_, map_res_;
  double max_vel_, max_acc_, dt_, pos_resolution_, vel_resolution_;
  int nx_, ny_, nv_, primitive_check_num_, max_expand_num_;
  double goal_tolerance_, safety_margin_, max_search_time_;
  Vec3 start_pos_, start_vel_, goal_pos_;

  int opt_iterations_, replan_iterations_, bspline_samples_per_segment_;
  double opt_step_, lambda_smooth_, lambda_collision_, d_safe_, cruise_speed_;

  double sensor_range_, sensor_fov_deg_, ray_step_, look_ahead_, last_heading_ = 0.0;

  double ctrl_dt_, elapsed_time_, total_time_, kp_, kd_, ctrl_max_acc_, ctrl_max_vel_;
  bool reached_goal_, planning_ok_; int step_count_;
  double replan_period_, last_replan_time_, wall_clock_;
  int replan_count_, full_replan_count_;
  double err_sum_sq_, err_max_; int err_count_; bool exec_collision_free_; int exec_collision_steps_;
  double current_error_ = 0.0; DesiredState current_desired_{};
  Vec3 robot_pos_{}, robot_vel_{};

  std::vector<BoxObstacle> true_obstacles_;
  std::vector<int8_t> occ_;
  std::vector<double> esdf_;
  int gnx_ = 0, gny_ = 0;

  std::vector<geometry_msgs::msg::Point> kino_path_, cps_, current_cps_, opt_path_, ref_pos_, actual_path_;
  std::vector<double> ref_time_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MappingDemoNode>());
  rclcpp::shutdown();
  return 0;
}
