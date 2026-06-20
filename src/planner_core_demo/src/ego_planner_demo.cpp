// ego_planner_demo.cpp
//
// Task 15-③: Reproduce the EGO-Planner core module and verify the computational
// efficiency gain of ESDF-free gradient-based planning.
//
// Idea reproduced here (Zhou et al., "EGO-Planner: An ESDF-free Gradient-based
// Local Planner for Quadrotor Fast Flight", RA-L 2020):
//   * A uniform cubic B-spline is parameterised by its control points Q_i.
//   * The trajectory is optimised by minimising
//         J = lambda_s * J_smooth + lambda_c * J_collision (+ lambda_f * J_feasibility)
//   * Fast-Planner builds an Euclidean Signed Distance Field (ESDF) and reads the
//     collision gradient from the field.  Building and querying the ESDF is the
//     dominant cost.
//   * EGO-Planner is "ESDF-free": for every control point that falls into a
//     collision it generates an anchor point p (on the obstacle surface) and a
//     repulsion direction v (pointing to free space).  The obstacle distance is
//     estimated analytically as d = (Q_i - p) . v, so the collision gradient is
//     -2 * lambda_c * (d_safe - d) * v with NO field lookup.
//
// This node runs BOTH back-ends on the SAME front-end path and reports the timing
// and trajectory-quality comparison, so the efficiency improvement is quantified.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

using namespace std::chrono_literals;

struct Vec3 { double x; double y; double z; };

struct BoxObstacle { double x; double y; double z; double sx; double sy; double sz; };

struct KinoNode
{
  Vec3 pos;
  Vec3 vel;
  double time;
  double g_score;
  double f_score;
  int parent;
  long long key;
};

struct QueueItem
{
  double f_score;
  int node_id;
  bool operator<(const QueueItem & other) const { return f_score > other.f_score; }
};

// Metrics describing one optimised trajectory.
struct BackendResult
{
  std::vector<geometry_msgs::msg::Point> path;
  double build_ms = 0.0;      // pre-computation (ESDF construction) time
  double opt_ms = 0.0;        // optimisation time
  double total_ms = 0.0;      // build + opt
  double length = 0.0;
  double smoothness = 0.0;
  double min_clearance = 0.0;
  bool collision_free = false;
};

class EgoPlannerDemoNode : public rclcpp::Node
{
public:
  EgoPlannerDemoNode() : Node("ego_planner_demo_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/markers", 10);

    initParams();
    generateObstacles();

    if (runKinoAStar()) {
      generateControlPointsFromKinoPath();
      esdf_result_ = optimizeEsdf();
      ego_result_ = optimizeEgo();
      reportComparison();
    } else {
      RCLCPP_ERROR(this->get_logger(), "Front-end Kinodynamic A* failed, skip back-ends.");
    }

    timer_ = this->create_wall_timer(
      500ms, std::bind(&EgoPlannerDemoNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "ego_planner_demo_node started.");
  }

private:
  // ------------------------------------------------------------------ params
  void initParams()
  {
    // Expanded arena: 16 x 14 x 4 m.
    x_min_ = -8.0; x_max_ = 8.0;
    y_min_ = -7.0; y_max_ = 7.0;
    z_min_ = 0.25; z_max_ = 4.25;

    pos_resolution_ = 0.4;
    vel_resolution_ = 0.5;
    nx_ = static_cast<int>((x_max_ - x_min_) / pos_resolution_);
    ny_ = static_cast<int>((y_max_ - y_min_) / pos_resolution_);
    nz_ = static_cast<int>((z_max_ - z_min_) / pos_resolution_);

    max_vel_ = 3.5;
    max_acc_ = 2.5;
    dt_ = 0.4;
    primitive_check_num_ = 8;
    goal_tolerance_ = 0.65;
    safety_margin_ = 0.18;
    max_search_time_ = 25.0;
    max_expand_num_ = 200000;

    start_pos_ = Vec3{-7.0, -6.0, 1.0};
    start_vel_ = Vec3{0.0, 0.0, 0.0};
    goal_pos_ = Vec3{7.0, 6.0, 1.0};
    nv_ = static_cast<int>(2.0 * max_vel_ / vel_resolution_) + 1;

    // Scenario type: "dense" | "clustered" | "narrow".
    scenario_ = this->declare_parameter<std::string>("scenario", "dense");

    opt_iterations_ = 300;
    opt_step_ = 0.06;
    lambda_smooth_ = 1.0;
    lambda_collision_ = 2.2;
    d_safe_ = 0.6;
    bspline_samples_per_segment_ = 12;

    esdf_resolution_ = 0.12;
  }

  void generateObstacles()
  {
    obstacles_.clear();
    const std::string scen = scenario_;
    std::mt19937 rng(12);

    if (scen == "narrow") {
      const double wall_z = 1.5, wall_h = 3.0;
      const double wall_span = 5.5, wall_thick = 0.35, gap_half = 1.2;
      auto addWall = [&](double cy, double gx) {
        obstacles_.push_back(BoxObstacle{-wall_span - 0.1, cy, wall_z,
                                          wall_span - gap_half + 0.1, wall_thick, wall_h});
        obstacles_.push_back(BoxObstacle{ gx + gap_half, cy, wall_z,
                                          wall_span - gap_half + 0.2, wall_thick, wall_h});
      };
      addWall(-1.5, -2.0); addWall(0.0, 1.5); addWall(1.3, -1.0);
      std::uniform_real_distribution<double> px(-6.5, 6.5), py(-5.8, 5.8), sz(0.4, 0.7), h(0.8, 2.2);
      for (int i = 0; i < 35; ++i) {
        BoxObstacle obs;
        obs.x = px(rng); obs.y = py(rng);
        obs.sx = sz(rng); obs.sy = sz(rng); obs.sz = h(rng); obs.z = obs.sz / 2.0;
        if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
        if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
        bool blocked = false;
        for (const auto & e : obstacles_)
          if (std::abs(obs.x - e.x) < e.sx / 2.0 + obs.sx / 2.0 + 0.3 &&
              std::abs(obs.y - e.y) < e.sy / 2.0 + obs.sy / 2.0 + 0.3) { blocked = true; break; }
        if (!blocked) obstacles_.push_back(obs);
      }
    } else if (scen == "clustered") {
      struct { double cx; double cy; int n; } clusters[] = {
        {-2.0, -2.0, 18}, {2.0, 1.5, 20}, {-0.5, 3.5, 16}
      };
      for (const auto & cl : clusters) {
        std::normal_distribution<double> cx(cl.cx, 2.0), cy(cl.cy, 1.8);
        std::uniform_real_distribution<double> sz(0.5, 0.95), h(0.9, 2.6);
        for (int i = 0; i < cl.n; ++i) {
          BoxObstacle obs;
          obs.x = cx(rng); obs.y = cy(rng);
          obs.sx = sz(rng); obs.sy = sz(rng); obs.sz = h(rng); obs.z = obs.sz / 2.0;
          if (std::abs(obs.x) > 7.5 || std::abs(obs.y) > 6.5) continue;
          if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
          if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
          obstacles_.push_back(obs);
        }
      }
    } else {
      // "dense" (default)
      std::uniform_real_distribution<double> pos_x(-7.2, 7.2), pos_y(-6.2, 6.2);
      std::uniform_real_distribution<double> size_xy(0.4, 0.85), height(0.9, 2.8);
      for (int i = 0; i < 40; ++i) {
        BoxObstacle obs;
        obs.x = pos_x(rng); obs.y = pos_y(rng);
        obs.sx = size_xy(rng); obs.sy = size_xy(rng);
        obs.sz = height(rng); obs.z = obs.sz / 2.0;
        if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
        if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
        obstacles_.push_back(obs);
      }
    }

    obstacles_.push_back(BoxObstacle{-1.2, -0.5, 1.6, 1.0, 2.8, 3.2});
    obstacles_.push_back(BoxObstacle{ 1.5,  1.2, 1.5, 2.4, 1.0, 3.0});
    obstacles_.push_back(BoxObstacle{ 2.5, -1.8, 1.3, 1.0, 2.6, 2.6});
  }

  // ------------------------------------------------------------------ helpers
  double distance2D(double x1, double y1, double x2, double y2) const
  {
    const double dx = x1 - x2, dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
  }
  double norm(const Vec3 & v) const { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
  double dist(const Vec3 & a, const Vec3 & b) const
  {
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  bool isInsideMap(const Vec3 & p) const
  {
    return p.x >= x_min_ && p.x < x_max_ && p.y >= y_min_ && p.y < y_max_ &&
           p.z >= z_min_ && p.z < z_max_;
  }
  bool isInsideObstacle(const Vec3 & p) const
  {
    for (const auto & obs : obstacles_) {
      if (std::abs(p.x - obs.x) <= obs.sx / 2.0 + safety_margin_ &&
          std::abs(p.y - obs.y) <= obs.sy / 2.0 + safety_margin_ &&
          std::abs(p.z - obs.z) <= obs.sz / 2.0 + safety_margin_) {
        return true;
      }
    }
    return false;
  }
  bool isStateValid(const Vec3 & p, const Vec3 & v) const
  {
    return isInsideMap(p) && !isInsideObstacle(p) && norm(v) <= max_vel_ + 1e-6;
  }
  double computeClearanceToBox(const Vec3 & p, const BoxObstacle & obs) const
  {
    const double dx = std::max(std::abs(p.x - obs.x) - obs.sx / 2.0, 0.0);
    const double dy = std::max(std::abs(p.y - obs.y) - obs.sy / 2.0, 0.0);
    const double dz = std::max(std::abs(p.z - obs.z) - obs.sz / 2.0, 0.0);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  // ------------------------------------------------------------- front end
  long long stateKey(const Vec3 & p, const Vec3 & v) const
  {
    int ix = static_cast<int>((p.x - x_min_) / pos_resolution_);
    int iy = static_cast<int>((p.y - y_min_) / pos_resolution_);
    int iz = static_cast<int>((p.z - z_min_) / pos_resolution_);
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) return -1;
    int ivx = static_cast<int>(std::round((v.x + max_vel_) / vel_resolution_));
    int ivy = static_cast<int>(std::round((v.y + max_vel_) / vel_resolution_));
    int ivz = static_cast<int>(std::round((v.z + max_vel_) / vel_resolution_));
    if (ivx < 0 || ivx >= nv_ || ivy < 0 || ivy >= nv_ || ivz < 0 || ivz >= nv_) return -1;
    long long key = ix;
    key = key * ny_ + iy; key = key * nz_ + iz;
    key = key * nv_ + ivx; key = key * nv_ + ivy; key = key * nv_ + ivz;
    return key;
  }
  Vec3 propagatePos(const Vec3 & p, const Vec3 & v, const Vec3 & a, double t) const
  {
    return Vec3{p.x + v.x * t + 0.5 * a.x * t * t,
                p.y + v.y * t + 0.5 * a.y * t * t,
                p.z + v.z * t + 0.5 * a.z * t * t};
  }
  Vec3 propagateVel(const Vec3 & v, const Vec3 & a, double t) const
  {
    return Vec3{v.x + a.x * t, v.y + a.y * t, v.z + a.z * t};
  }
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
    std::vector<Vec3> controls;
    const std::array<double, 3> values = {-max_acc_, 0.0, max_acc_};
    for (double ax : values)
      for (double ay : values)
        for (double az : values) {
          Vec3 a{ax, ay, az};
          if (norm(a) <= max_acc_ * 1.75) controls.push_back(a);
        }
    return controls;
  }
  double heuristic(const Vec3 & p, const Vec3 & v) const
  {
    const double d = dist(p, goal_pos_);
    Vec3 to_goal{goal_pos_.x - p.x, goal_pos_.y - p.y, goal_pos_.z - p.z};
    const double dn = std::max(d, 1e-3);
    Vec3 dv{max_vel_ * to_goal.x / dn, max_vel_ * to_goal.y / dn, max_vel_ * to_goal.z / dn};
    return d / max_vel_ + 0.08 * dist(v, dv);
  }

  bool runKinoAStar()
  {
    if (!isStateValid(start_pos_, start_vel_)) return false;
    const auto controls = generateControlSamples();
    std::vector<KinoNode> nodes;
    std::priority_queue<QueueItem> open_set;
    std::unordered_map<long long, double> best_g;
    std::unordered_map<long long, int> best_id;
    std::unordered_set<long long> closed;

    long long start_key = stateKey(start_pos_, start_vel_);
    KinoNode s; s.pos = start_pos_; s.vel = start_vel_; s.time = 0.0;
    s.g_score = 0.0; s.f_score = heuristic(start_pos_, start_vel_); s.parent = -1; s.key = start_key;
    nodes.push_back(s); best_g[start_key] = 0.0; best_id[start_key] = 0;
    open_set.push(QueueItem{s.f_score, 0});

    int final_id = -1, expand_count = 0;
    while (!open_set.empty() && expand_count < max_expand_num_) {
      QueueItem item = open_set.top(); open_set.pop();
      const KinoNode current = nodes[item.node_id];
      auto bit = best_id.find(current.key);
      if (bit == best_id.end() || bit->second != item.node_id) continue;
      if (closed.count(current.key)) continue;
      closed.insert(current.key); expand_count++;
      if (dist(current.pos, goal_pos_) < goal_tolerance_ && current.time > 1.0) { final_id = item.node_id; break; }
      if (current.time > max_search_time_) continue;
      for (const auto & acc : controls) {
        if (!checkPrimitiveCollision(current.pos, current.vel, acc)) continue;
        Vec3 np = propagatePos(current.pos, current.vel, acc, dt_);
        Vec3 nvv = propagateVel(current.vel, acc, dt_);
        long long nk = stateKey(np, nvv);
        if (nk < 0 || closed.count(nk)) continue;
        const double ng = current.g_score + dt_ + 0.03 * norm(acc) * norm(acc) * dt_;
        auto git = best_g.find(nk);
        if (git == best_g.end() || ng < git->second) {
          KinoNode n; n.pos = np; n.vel = nvv; n.time = current.time + dt_;
          n.g_score = ng; n.f_score = ng + heuristic(np, nvv); n.parent = item.node_id; n.key = nk;
          int id = static_cast<int>(nodes.size()); nodes.push_back(n);
          best_g[nk] = ng; best_id[nk] = id; open_set.push(QueueItem{n.f_score, id});
        }
      }
    }
    kino_path_.clear();
    if (final_id < 0) return false;
    std::vector<int> ids; int id = final_id;
    while (id >= 0) { ids.push_back(id); id = nodes[id].parent; }
    std::reverse(ids.begin(), ids.end());
    for (int nid : ids) {
      geometry_msgs::msg::Point p;
      p.x = nodes[nid].pos.x; p.y = nodes[nid].pos.y; p.z = nodes[nid].pos.z;
      kino_path_.push_back(p);
    }
    RCLCPP_INFO(this->get_logger(), "Front-end ready: kino path with %zu points.", kino_path_.size());
    return true;
  }

  // ------------------------------------------------------- B-spline plumbing
  void generateControlPointsFromKinoPath()
  {
    init_control_points_.clear();
    if (kino_path_.size() < 2) return;
    // Clamp the curve at both ends by repeating the first/last point (cubic B-spline).
    init_control_points_.push_back(kino_path_.front());
    init_control_points_.push_back(kino_path_.front());
    for (const auto & p : kino_path_) init_control_points_.push_back(p);
    init_control_points_.push_back(kino_path_.back());
    init_control_points_.push_back(kino_path_.back());
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
    p.z = b0 * p0.z + b1 * p1.z + b2 * p2.z + b3 * p3.z;
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

  // Smoothness (elastic / second-difference) gradient, shared by both back-ends.
  void addSmoothnessGradient(
    const std::vector<geometry_msgs::msg::Point> & q, std::vector<Vec3> & grad) const
  {
    // J_s = sum_i || q_{i+1} - 2 q_i + q_{i-1} ||^2
    for (size_t i = 1; i + 1 < q.size(); ++i) {
      Vec3 a{q[i + 1].x - 2.0 * q[i].x + q[i - 1].x,
             q[i + 1].y - 2.0 * q[i].y + q[i - 1].y,
             q[i + 1].z - 2.0 * q[i].z + q[i - 1].z};
      // d/dq_{i-1} : +2a ; d/dq_i : -4a ; d/dq_{i+1} : +2a
      grad[i - 1].x += lambda_smooth_ * 2.0 * a.x;
      grad[i - 1].y += lambda_smooth_ * 2.0 * a.y;
      grad[i - 1].z += lambda_smooth_ * 2.0 * a.z;
      grad[i].x     -= lambda_smooth_ * 4.0 * a.x;
      grad[i].y     -= lambda_smooth_ * 4.0 * a.y;
      grad[i].z     -= lambda_smooth_ * 4.0 * a.z;
      grad[i + 1].x += lambda_smooth_ * 2.0 * a.x;
      grad[i + 1].y += lambda_smooth_ * 2.0 * a.y;
      grad[i + 1].z += lambda_smooth_ * 2.0 * a.z;
    }
  }

  // ====================================================================== //
  //  BACK-END A : ESDF-based optimisation (Fast-Planner style)            //
  // ====================================================================== //
  void buildEsdf()
  {
    esdf_nx_ = static_cast<int>((x_max_ - x_min_) / esdf_resolution_) + 1;
    esdf_ny_ = static_cast<int>((y_max_ - y_min_) / esdf_resolution_) + 1;
    esdf_nz_ = static_cast<int>((z_max_ - z_min_) / esdf_resolution_) + 1;
    esdf_.assign(static_cast<size_t>(esdf_nx_) * esdf_ny_ * esdf_nz_, 0.0);

    // For every grid cell store the Euclidean distance to the nearest obstacle
    // surface.  This is exactly the pre-computation that EGO-Planner removes.
    for (int ix = 0; ix < esdf_nx_; ++ix) {
      const double x = x_min_ + ix * esdf_resolution_;
      for (int iy = 0; iy < esdf_ny_; ++iy) {
        const double y = y_min_ + iy * esdf_resolution_;
        for (int iz = 0; iz < esdf_nz_; ++iz) {
          const double z = z_min_ + iz * esdf_resolution_;
          const Vec3 p{x, y, z};
          double best = std::numeric_limits<double>::infinity();
          for (const auto & obs : obstacles_)
            best = std::min(best, computeClearanceToBox(p, obs));
          esdf_[esdfIndex(ix, iy, iz)] = best;
        }
      }
    }
  }
  size_t esdfIndex(int ix, int iy, int iz) const
  {
    return (static_cast<size_t>(ix) * esdf_ny_ + iy) * esdf_nz_ + iz;
  }
  double esdfAt(const Vec3 & p) const
  {
    int ix = std::clamp(static_cast<int>((p.x - x_min_) / esdf_resolution_), 0, esdf_nx_ - 1);
    int iy = std::clamp(static_cast<int>((p.y - y_min_) / esdf_resolution_), 0, esdf_ny_ - 1);
    int iz = std::clamp(static_cast<int>((p.z - z_min_) / esdf_resolution_), 0, esdf_nz_ - 1);
    return esdf_[esdfIndex(ix, iy, iz)];
  }
  // Collision gradient obtained by finite-differencing the ESDF (field lookup).
  Vec3 esdfCollisionGradient(const geometry_msgs::msg::Point & pt) const
  {
    const Vec3 p{pt.x, pt.y, pt.z};
    const double d = esdfAt(p);
    if (d >= d_safe_) return Vec3{0.0, 0.0, 0.0};
    const double h = esdf_resolution_;
    Vec3 g{
      (esdfAt(Vec3{p.x + h, p.y, p.z}) - esdfAt(Vec3{p.x - h, p.y, p.z})) / (2.0 * h),
      (esdfAt(Vec3{p.x, p.y + h, p.z}) - esdfAt(Vec3{p.x, p.y - h, p.z})) / (2.0 * h),
      (esdfAt(Vec3{p.x, p.y, p.z + h}) - esdfAt(Vec3{p.x, p.y, p.z - h})) / (2.0 * h)};
    // dJ/dq = -2 lambda_c (d_safe - d) * grad(d)
    const double coef = -2.0 * lambda_collision_ * (d_safe_ - d);
    return Vec3{coef * g.x, coef * g.y, coef * g.z};
  }

  BackendResult optimizeEsdf()
  {
    BackendResult r;
    auto t_build0 = std::chrono::steady_clock::now();
    buildEsdf();
    auto t_build1 = std::chrono::steady_clock::now();
    r.build_ms = std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();

    auto q = init_control_points_;
    auto t_opt0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < opt_iterations_; ++iter) {
      std::vector<Vec3> grad(q.size(), Vec3{0.0, 0.0, 0.0});
      addSmoothnessGradient(q, grad);
      for (size_t i = 0; i < q.size(); ++i) {
        Vec3 g = esdfCollisionGradient(q[i]);
        grad[i].x += g.x; grad[i].y += g.y; grad[i].z += g.z;
      }
      applyGradientStep(q, grad);
    }
    auto t_opt1 = std::chrono::steady_clock::now();
    r.opt_ms = std::chrono::duration<double, std::milli>(t_opt1 - t_opt0).count();
    r.total_ms = r.build_ms + r.opt_ms;
    finalizeResult(q, r);
    return r;
  }

  // ====================================================================== //
  //  BACK-END B : EGO-Planner ESDF-free optimisation                     //
  // ====================================================================== //
  // For a control point inside the danger zone, find the nearest obstacle, its
  // surface anchor p and the unit repulsion direction v.  Distance estimate is
  // d = (q - p) . v ; gradient is -2 lambda_c (d_safe - d) * v.  No field used.
  Vec3 egoCollisionGradient(const geometry_msgs::msg::Point & pt) const
  {
    const Vec3 q{pt.x, pt.y, pt.z};
    double best = std::numeric_limits<double>::infinity();
    const BoxObstacle * best_obs = nullptr;
    for (const auto & obs : obstacles_) {
      const double c = computeClearanceToBox(q, obs);
      if (c < best) { best = c; best_obs = &obs; }
    }
    if (best_obs == nullptr || best >= d_safe_) return Vec3{0.0, 0.0, 0.0};

    // Surface anchor: closest point on the (inflated) box to q.
    const double hx = best_obs->sx / 2.0, hy = best_obs->sy / 2.0, hz = best_obs->sz / 2.0;
    const double cx = std::clamp(q.x, best_obs->x - hx, best_obs->x + hx);
    const double cy = std::clamp(q.y, best_obs->y - hy, best_obs->y + hy);
    const double cz = std::clamp(q.z, best_obs->z - hz, best_obs->z + hz);
    Vec3 v{q.x - cx, q.y - cy, q.z - cz};
    double n = norm(v);
    if (n < 1e-6) {
      // q is inside the box: push it out through the closest face.
      const double dxf = hx - std::abs(q.x - best_obs->x);
      const double dyf = hy - std::abs(q.y - best_obs->y);
      const double dzf = hz - std::abs(q.z - best_obs->z);
      if (dxf <= dyf && dxf <= dzf) v = Vec3{(q.x >= best_obs->x) ? 1.0 : -1.0, 0, 0};
      else if (dyf <= dzf)         v = Vec3{0, (q.y >= best_obs->y) ? 1.0 : -1.0, 0};
      else                          v = Vec3{0, 0, (q.z >= best_obs->z) ? 1.0 : -1.0};
      n = 1.0;
    }
    v.x /= n; v.y /= n; v.z /= n;
    const double d = best;               // d = (q - p) . v == clearance
    const double coef = -2.0 * lambda_collision_ * (d_safe_ - d);
    return Vec3{coef * v.x, coef * v.y, coef * v.z};
  }

  BackendResult optimizeEgo()
  {
    BackendResult r;
    r.build_ms = 0.0;        // EGO-Planner needs NO field pre-computation.
    auto q = init_control_points_;
    auto t_opt0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < opt_iterations_; ++iter) {
      std::vector<Vec3> grad(q.size(), Vec3{0.0, 0.0, 0.0});
      addSmoothnessGradient(q, grad);
      for (size_t i = 0; i < q.size(); ++i) {
        Vec3 g = egoCollisionGradient(q[i]);
        grad[i].x += g.x; grad[i].y += g.y; grad[i].z += g.z;
      }
      applyGradientStep(q, grad);
    }
    auto t_opt1 = std::chrono::steady_clock::now();
    r.opt_ms = std::chrono::duration<double, std::milli>(t_opt1 - t_opt0).count();
    r.total_ms = r.build_ms + r.opt_ms;
    finalizeResult(q, r);
    return r;
  }

  // ------------------------------------------------------- optimiser shared
  void applyGradientStep(std::vector<geometry_msgs::msg::Point> & q,
                         const std::vector<Vec3> & grad) const
  {
    // Gradient descent.  The first/last two clamped control points stay fixed so
    // the trajectory keeps its boundary conditions.
    for (size_t i = 2; i + 2 < q.size(); ++i) {
      q[i].x -= opt_step_ * grad[i].x;
      q[i].y -= opt_step_ * grad[i].y;
      q[i].z -= opt_step_ * grad[i].z;
      q[i].x = std::clamp(q[i].x, x_min_ + 0.2, x_max_ - 0.2);
      q[i].y = std::clamp(q[i].y, y_min_ + 0.2, y_max_ - 0.2);
      q[i].z = std::clamp(q[i].z, z_min_ + 0.2, z_max_ - 0.2);
    }
  }

  void finalizeResult(const std::vector<geometry_msgs::msg::Point> & q, BackendResult & r) const
  {
    r.path = sampleBspline(q);
    r.length = computePathLength(r.path);
    r.smoothness = computeSmoothnessCost(r.path);
    r.min_clearance = computeMinClearance(r.path);
    r.collision_free = checkCollisionFree(r.path);
  }

  double computePathLength(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    double l = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
      const double dx = path[i].x - path[i - 1].x, dy = path[i].y - path[i - 1].y,
                   dz = path[i].z - path[i - 1].z;
      l += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return l;
  }
  double computeSmoothnessCost(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    if (path.size() < 3) return 0.0;
    double c = 0.0;
    for (size_t i = 1; i + 1 < path.size(); ++i) {
      const double ax = path[i + 1].x - 2.0 * path[i].x + path[i - 1].x;
      const double ay = path[i + 1].y - 2.0 * path[i].y + path[i - 1].y;
      const double az = path[i + 1].z - 2.0 * path[i].z + path[i - 1].z;
      c += ax * ax + ay * ay + az * az;
    }
    return c;
  }
  double computeMinClearance(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    if (path.empty() || obstacles_.empty()) return 0.0;
    double m = std::numeric_limits<double>::infinity();
    for (const auto & pt : path) {
      const Vec3 p{pt.x, pt.y, pt.z};
      for (const auto & obs : obstacles_) m = std::min(m, computeClearanceToBox(p, obs));
    }
    return std::isfinite(m) ? m : 0.0;
  }
  bool checkCollisionFree(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    for (const auto & pt : path) {
      const Vec3 p{pt.x, pt.y, pt.z};
      if (!isInsideMap(p) || isInsideObstacle(p)) return false;
    }
    return true;
  }

  void reportComparison()
  {
    const double speedup = (ego_result_.total_ms > 1e-6)
      ? esdf_result_.total_ms / ego_result_.total_ms : 0.0;
    RCLCPP_INFO(this->get_logger(), "================ EGO vs ESDF back-end ================");
    RCLCPP_INFO(this->get_logger(),
      "ESDF  | build_ms: %.3f | opt_ms: %.3f | total_ms: %.3f | len: %.3f | smooth: %.4f | clr: %.3f | free: %s",
      esdf_result_.build_ms, esdf_result_.opt_ms, esdf_result_.total_ms,
      esdf_result_.length, esdf_result_.smoothness, esdf_result_.min_clearance,
      esdf_result_.collision_free ? "true" : "false");
    RCLCPP_INFO(this->get_logger(),
      "EGO   | build_ms: %.3f | opt_ms: %.3f | total_ms: %.3f | len: %.3f | smooth: %.4f | clr: %.3f | free: %s",
      ego_result_.build_ms, ego_result_.opt_ms, ego_result_.total_ms,
      ego_result_.length, ego_result_.smoothness, ego_result_.min_clearance,
      ego_result_.collision_free ? "true" : "false");
    RCLCPP_INFO(this->get_logger(),
      "ESDF grid: %d x %d x %d = %zu cells (built every plan). EGO needs none.",
      esdf_nx_, esdf_ny_, esdf_nz_,
      static_cast<size_t>(esdf_nx_) * esdf_ny_ * esdf_nz_);
    RCLCPP_INFO(this->get_logger(),
      ">>> EGO total speed-up vs ESDF: %.2fx (ESDF build overhead avoided) <<<", speedup);
    RCLCPP_INFO(this->get_logger(), "======================================================");
  }

  // ------------------------------------------------------------- visualise
  void timerCallback()
  {
    visualization_msgs::msg::MarkerArray array;
    addObstacleMarkers(array);
    addStartGoalMarkers(array);
    addLineMarker(array, "kino_astar_path", 2000, kino_path_, 0.07, 0.1, 0.9, 0.1);
    addLineMarker(array, "esdf_path", 3000, esdf_result_.path, 0.055, 1.0, 0.55, 0.0);
    addLineMarker(array, "ego_path", 3100, ego_result_.path, 0.055, 0.2, 0.6, 1.0);
    marker_pub_->publish(array);
  }
  void addObstacleMarkers(visualization_msgs::msg::MarkerArray & array)
  {
    int id = 0;
    for (const auto & obs : obstacles_) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map"; m.header.stamp = this->now();
      m.ns = "obstacles"; m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = obs.x; m.pose.position.y = obs.y; m.pose.position.z = obs.z;
      m.pose.orientation.w = 1.0;
      m.scale.x = obs.sx; m.scale.y = obs.sy; m.scale.z = obs.sz;
      m.color.r = 0.8; m.color.g = 0.1; m.color.b = 0.1; m.color.a = 0.7;
      array.markers.push_back(m);
    }
  }
  void addStartGoalMarkers(visualization_msgs::msg::MarkerArray & array)
  {
    for (int k = 0; k < 2; ++k) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map"; m.header.stamp = this->now();
      m.ns = "start_goal"; m.id = 1000 + k;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      const Vec3 & p = (k == 0) ? start_pos_ : goal_pos_;
      m.pose.position.x = p.x; m.pose.position.y = p.y; m.pose.position.z = p.z;
      m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = 0.35;
      m.color.r = (k == 0) ? 0.0 : 0.0; m.color.g = (k == 0) ? 1.0 : 0.2;
      m.color.b = (k == 0) ? 0.0 : 1.0; m.color.a = 1.0;
      array.markers.push_back(m);
    }
  }
  void addLineMarker(visualization_msgs::msg::MarkerArray & array, const std::string & ns,
                     int id, const std::vector<geometry_msgs::msg::Point> & pts,
                     double w, double r, double g, double b)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = ns; m.id = id;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = w; m.pose.orientation.w = 1.0;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;
    m.points = pts;
    array.markers.push_back(m);
  }

  // ------------------------------------------------------------------ data
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  double pos_resolution_, vel_resolution_;
  int nx_, ny_, nz_, nv_;
  double max_vel_, max_acc_, dt_;
  int primitive_check_num_;
  double goal_tolerance_, safety_margin_, max_search_time_;
  int max_expand_num_;
  Vec3 start_pos_, start_vel_, goal_pos_;

  int opt_iterations_;
  double opt_step_, lambda_smooth_, lambda_collision_, d_safe_;
  int bspline_samples_per_segment_;
  double esdf_resolution_;

  std::string scenario_;

  std::vector<BoxObstacle> obstacles_;
  std::vector<geometry_msgs::msg::Point> kino_path_;
  std::vector<geometry_msgs::msg::Point> init_control_points_;

  std::vector<double> esdf_;
  int esdf_nx_ = 0, esdf_ny_ = 0, esdf_nz_ = 0;

  BackendResult esdf_result_;
  BackendResult ego_result_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EgoPlannerDemoNode>());
  rclcpp::shutdown();
  return 0;
}
