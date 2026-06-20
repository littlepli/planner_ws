// bspline_backend_demo.cpp
//
// Task 15-②: Reproduce the Fast-Planner back-end = uniform cubic B-spline
// trajectory optimisation driven by an Euclidean Signed Distance Field (ESDF).
//
// Pipeline:
//   1. Kinodynamic A* front-end produces an initial path.
//   2. The path becomes the control points of a clamped uniform cubic B-spline.
//   3. A real ESDF is built over the map (signed distance to the nearest inflated
//      obstacle, stored on a grid, queried by trilinear interpolation).
//   4. The control points are optimised by gradient descent minimising
//          J = lambda_s * J_smooth        (elastic / second-difference energy)
//            + lambda_c * J_collision     (ESDF gradient pushes the curve clear)
//            + lambda_f * J_feasibility   (B-spline velocity / acceleration limits)
//   5. The trajectory smoothness and safety BEFORE and AFTER optimisation are
//      reported so the effect of the ESDF gradient optimisation is quantified.
//
// This is the genuine ESDF-gradient back-end (compare with ego_planner_demo, which
// removes the ESDF and obtains the same gradient analytically).

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
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

using namespace std::chrono_literals;

struct Vec3 { double x; double y; double z; };
struct BoxObstacle { double x; double y; double z; double sx; double sy; double sz; };
struct KinoNode { Vec3 pos; Vec3 vel; double time; double g_score; double f_score; int parent; long long key; };
struct QueueItem { double f_score; int node_id; bool operator<(const QueueItem & o) const { return f_score > o.f_score; } };

class BsplineBackendDemoNode : public rclcpp::Node
{
public:
  BsplineBackendDemoNode() : Node("bspline_backend_demo_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/markers", 10);

    initParams();
    generateObstacles();

    if (runKinoAStar()) {
      runBsplineBackend();
    } else {
      RCLCPP_ERROR(this->get_logger(), "Skip B-spline backend because Kinodynamic A* failed.");
    }

    timer_ = this->create_wall_timer(500ms, std::bind(&BsplineBackendDemoNode::timerCallback, this));
    RCLCPP_INFO(this->get_logger(), "bspline_backend_demo_node started.");
  }

private:
  // ------------------------------------------------------------------ params
  void initParams()
  {
    // Expanded arena: 16 x 14 x 4 m.
    x_min_ = -8.0; x_max_ = 8.0; y_min_ = -7.0; y_max_ = 7.0; z_min_ = 0.25; z_max_ = 4.25;
    pos_resolution_ = 0.4; vel_resolution_ = 0.5;
    nx_ = static_cast<int>((x_max_ - x_min_) / pos_resolution_);
    ny_ = static_cast<int>((y_max_ - y_min_) / pos_resolution_);
    nz_ = static_cast<int>((z_max_ - z_min_) / pos_resolution_);
    max_vel_ = 3.5; max_acc_ = 2.5; dt_ = 0.4;
    primitive_check_num_ = 8; goal_tolerance_ = 0.65; safety_margin_ = 0.18;
    max_search_time_ = 25.0; max_expand_num_ = 200000;
    start_pos_ = Vec3{-7.0, -6.0, 1.0}; start_vel_ = Vec3{0.0, 0.0, 0.0}; goal_pos_ = Vec3{7.0, 6.0, 1.0};
    nv_ = static_cast<int>(2.0 * max_vel_ / vel_resolution_) + 1;

    // Scenario type.
    scenario_ = this->declare_parameter<std::string>("scenario", "dense");

    bspline_samples_per_segment_ = 12;
    knot_dt_ = dt_;

    esdf_resolution_ = 0.10;
    opt_iterations_ = 500;
    opt_step_ = 0.08;
    max_move_ = 0.03;
    lambda_smooth_ = 1.0;
    lambda_collision_ = 2.5;
    lambda_feasible_ = 0.12;
    d_safe_ = 0.55;
  }

  void generateObstacles()
  {
    obstacles_.clear();
    const std::string scen = scenario_;
    std::mt19937 rng(12);

    if (scen == "narrow") {
      const double wall_z = 1.5, wall_h = 3.0, wall_span = 5.5, wall_thick = 0.35, gap_half = 1.2;
      auto addWall = [&](double cy, double gx) {
        obstacles_.push_back(BoxObstacle{-wall_span - 0.1, cy, wall_z, wall_span - gap_half + 0.1, wall_thick, wall_h});
        obstacles_.push_back(BoxObstacle{ gx + gap_half, cy, wall_z, wall_span - gap_half + 0.2, wall_thick, wall_h});
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
      struct { double cx; double cy; int n; } clusters[] = { {-2.0, -2.0, 18}, {2.0, 1.5, 20}, {-0.5, 3.5, 16} };
      for (const auto & cl : clusters) {
        std::normal_distribution<double> cx(cl.cx, 2.0), cy(cl.cy, 1.8);
        std::uniform_real_distribution<double> sz(0.5, 0.95), h(0.9, 2.6);
        for (int i = 0; i < cl.n; ++i) {
          BoxObstacle obs;
          obs.x = cx(rng); obs.y = cy(rng); obs.sx = sz(rng); obs.sy = sz(rng); obs.sz = h(rng); obs.z = obs.sz / 2.0;
          if (std::abs(obs.x) > 7.5 || std::abs(obs.y) > 6.5) continue;
          if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
          if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
          obstacles_.push_back(obs);
        }
      }
    } else {
      std::uniform_real_distribution<double> pos_x(-7.2, 7.2), pos_y(-6.2, 6.2);
      std::uniform_real_distribution<double> size_xy(0.4, 0.85), height(0.9, 2.8);
      for (int i = 0; i < 40; ++i) {
        BoxObstacle obs;
        obs.x = pos_x(rng); obs.y = pos_y(rng);
        obs.sx = size_xy(rng); obs.sy = size_xy(rng); obs.sz = height(rng); obs.z = obs.sz / 2.0;
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
  { const double dx = x1 - x2, dy = y1 - y2; return std::sqrt(dx * dx + dy * dy); }
  double norm(const Vec3 & v) const { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
  double dist(const Vec3 & a, const Vec3 & b) const
  { const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z; return std::sqrt(dx * dx + dy * dy + dz * dz); }
  bool isInsideMap(const Vec3 & p) const
  { return p.x >= x_min_ && p.x < x_max_ && p.y >= y_min_ && p.y < y_max_ && p.z >= z_min_ && p.z < z_max_; }
  bool isInsideObstacle(const Vec3 & p) const
  {
    for (const auto & obs : obstacles_)
      if (std::abs(p.x - obs.x) <= obs.sx / 2.0 + safety_margin_ &&
          std::abs(p.y - obs.y) <= obs.sy / 2.0 + safety_margin_ &&
          std::abs(p.z - obs.z) <= obs.sz / 2.0 + safety_margin_) return true;
    return false;
  }
  bool isStateValid(const Vec3 & p, const Vec3 & v) const
  { return isInsideMap(p) && !isInsideObstacle(p) && norm(v) <= max_vel_ + 1e-6; }
  double computeClearanceToBox(const Vec3 & p, const BoxObstacle & obs) const
  {
    const double dx = std::max(std::abs(p.x - obs.x) - obs.sx / 2.0, 0.0);
    const double dy = std::max(std::abs(p.y - obs.y) - obs.sy / 2.0, 0.0);
    const double dz = std::max(std::abs(p.z - obs.z) - obs.sz / 2.0, 0.0);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  // =================================================================== //
  //  ESDF : signed distance to the nearest inflated obstacle on a grid  //
  // =================================================================== //
  // Signed distance from p to one axis-aligned box inflated by safety_margin.
  double boxSignedDistance(const Vec3 & p, const BoxObstacle & obs) const
  {
    const double hx = obs.sx / 2.0 + safety_margin_;
    const double hy = obs.sy / 2.0 + safety_margin_;
    const double hz = obs.sz / 2.0 + safety_margin_;
    const double qx = std::abs(p.x - obs.x) - hx;
    const double qy = std::abs(p.y - obs.y) - hy;
    const double qz = std::abs(p.z - obs.z) - hz;
    const double ox = std::max(qx, 0.0), oy = std::max(qy, 0.0), oz = std::max(qz, 0.0);
    const double outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const double inside = std::min(std::max(qx, std::max(qy, qz)), 0.0);
    return outside + inside;       // >0 outside, <0 inside the inflated box
  }
  // Signed distance to the obstacle field = min over all boxes.
  double sceneSignedDistance(const Vec3 & p) const
  {
    double d = std::numeric_limits<double>::infinity();
    for (const auto & obs : obstacles_) d = std::min(d, boxSignedDistance(p, obs));
    return std::isfinite(d) ? d : 10.0;
  }
  size_t esdfIndex(int ix, int iy, int iz) const
  { return (static_cast<size_t>(ix) * esdf_ny_ + iy) * esdf_nz_ + iz; }

  void buildEsdf()
  {
    esdf_nx_ = static_cast<int>((x_max_ - x_min_) / esdf_resolution_) + 1;
    esdf_ny_ = static_cast<int>((y_max_ - y_min_) / esdf_resolution_) + 1;
    esdf_nz_ = static_cast<int>((z_max_ - z_min_) / esdf_resolution_) + 1;
    esdf_.assign(static_cast<size_t>(esdf_nx_) * esdf_ny_ * esdf_nz_, 0.0);
    for (int ix = 0; ix < esdf_nx_; ++ix) {
      const double x = x_min_ + ix * esdf_resolution_;
      for (int iy = 0; iy < esdf_ny_; ++iy) {
        const double y = y_min_ + iy * esdf_resolution_;
        for (int iz = 0; iz < esdf_nz_; ++iz) {
          const double z = z_min_ + iz * esdf_resolution_;
          esdf_[esdfIndex(ix, iy, iz)] = sceneSignedDistance(Vec3{x, y, z});
        }
      }
    }
  }
  // Trilinear interpolation of the stored ESDF (this is the field "query").
  double esdfValue(const Vec3 & p) const
  {
    const double fx = (p.x - x_min_) / esdf_resolution_;
    const double fy = (p.y - y_min_) / esdf_resolution_;
    const double fz = (p.z - z_min_) / esdf_resolution_;
    int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, esdf_nx_ - 2);
    int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, esdf_ny_ - 2);
    int iz = std::clamp(static_cast<int>(std::floor(fz)), 0, esdf_nz_ - 2);
    const double tx = std::clamp(fx - ix, 0.0, 1.0);
    const double ty = std::clamp(fy - iy, 0.0, 1.0);
    const double tz = std::clamp(fz - iz, 0.0, 1.0);
    const double c000 = esdf_[esdfIndex(ix, iy, iz)];
    const double c100 = esdf_[esdfIndex(ix + 1, iy, iz)];
    const double c010 = esdf_[esdfIndex(ix, iy + 1, iz)];
    const double c110 = esdf_[esdfIndex(ix + 1, iy + 1, iz)];
    const double c001 = esdf_[esdfIndex(ix, iy, iz + 1)];
    const double c101 = esdf_[esdfIndex(ix + 1, iy, iz + 1)];
    const double c011 = esdf_[esdfIndex(ix, iy + 1, iz + 1)];
    const double c111 = esdf_[esdfIndex(ix + 1, iy + 1, iz + 1)];
    const double c00 = c000 * (1 - tx) + c100 * tx;
    const double c10 = c010 * (1 - tx) + c110 * tx;
    const double c01 = c001 * (1 - tx) + c101 * tx;
    const double c11 = c011 * (1 - tx) + c111 * tx;
    const double c0 = c00 * (1 - ty) + c10 * ty;
    const double c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
  }
  // Distance value + gradient (central difference on the interpolated field).
  double esdfValueAndGradient(const Vec3 & p, Vec3 & grad) const
  {
    const double h = esdf_resolution_;
    grad.x = (esdfValue(Vec3{p.x + h, p.y, p.z}) - esdfValue(Vec3{p.x - h, p.y, p.z})) / (2 * h);
    grad.y = (esdfValue(Vec3{p.x, p.y + h, p.z}) - esdfValue(Vec3{p.x, p.y - h, p.z})) / (2 * h);
    grad.z = (esdfValue(Vec3{p.x, p.y, p.z + h}) - esdfValue(Vec3{p.x, p.y, p.z - h})) / (2 * h);
    return esdfValue(p);
  }

  // ============================ FRONT-END ============================= //
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
    long long key = ix; key = key * ny_ + iy; key = key * nz_ + iz;
    key = key * nv_ + ivx; key = key * nv_ + ivy; key = key * nv_ + ivz;
    return key;
  }
  Vec3 propagatePos(const Vec3 & p, const Vec3 & v, const Vec3 & a, double t) const
  { return Vec3{p.x + v.x * t + 0.5 * a.x * t * t, p.y + v.y * t + 0.5 * a.y * t * t, p.z + v.z * t + 0.5 * a.z * t * t}; }
  Vec3 propagateVel(const Vec3 & v, const Vec3 & a, double t) const
  { return Vec3{v.x + a.x * t, v.y + a.y * t, v.z + a.z * t}; }
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
    const std::array<double, 3> values = {-max_acc_, 0.0, max_acc_};
    for (double ax : values) for (double ay : values) for (double az : values) {
      Vec3 a{ax, ay, az}; if (norm(a) <= max_acc_ * 1.75) c.push_back(a);
    }
    return c;
  }
  double heuristic(const Vec3 & p, const Vec3 & v) const
  {
    const double d = dist(p, goal_pos_);
    Vec3 tg{goal_pos_.x - p.x, goal_pos_.y - p.y, goal_pos_.z - p.z};
    const double dn = std::max(d, 1e-3);
    Vec3 dv{max_vel_ * tg.x / dn, max_vel_ * tg.y / dn, max_vel_ * tg.z / dn};
    return d / max_vel_ + 0.08 * dist(v, dv);
  }
  bool runKinoAStar()
  {
    const auto t0 = std::chrono::steady_clock::now();
    if (!isStateValid(start_pos_, start_vel_)) return false;
    const auto controls = generateControlSamples();
    std::vector<KinoNode> nodes;
    std::priority_queue<QueueItem> open_set;
    std::unordered_map<long long, double> best_g;
    std::unordered_map<long long, int> best_id;
    std::unordered_set<long long> closed;
    long long sk = stateKey(start_pos_, start_vel_);
    KinoNode s; s.pos = start_pos_; s.vel = start_vel_; s.time = 0.0;
    s.g_score = 0.0; s.f_score = heuristic(start_pos_, start_vel_); s.parent = -1; s.key = sk;
    nodes.push_back(s); best_g[sk] = 0.0; best_id[sk] = 0; open_set.push(QueueItem{s.f_score, 0});
    int final_id = -1, expand = 0;
    while (!open_set.empty() && expand < max_expand_num_) {
      QueueItem it = open_set.top(); open_set.pop();
      const KinoNode cur = nodes[it.node_id];
      auto bit = best_id.find(cur.key);
      if (bit == best_id.end() || bit->second != it.node_id) continue;
      if (closed.count(cur.key)) continue;
      closed.insert(cur.key); expand++;
      if (dist(cur.pos, goal_pos_) < goal_tolerance_ && cur.time > 1.0) { final_id = it.node_id; break; }
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
    const auto t1 = std::chrono::steady_clock::now();
    frontend_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    traj_duration_ = nodes[final_id].time;
    RCLCPP_INFO(this->get_logger(),
      "Kinodynamic A* success. nodes: %d, path points: %zu, time: %.1f ms",
      expand, kino_path_.size(), frontend_ms_);
    return true;
  }

  // ============================ B-SPLINE ============================== //
  void generateControlPointsFromKinoPath()
  {
    init_control_points_.clear();
    if (kino_path_.size() < 2) return;
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

  // -------- the three cost gradients of the Fast-Planner back-end --------
  void addSmoothnessGradient(const std::vector<geometry_msgs::msg::Point> & q,
                             std::vector<Vec3> & grad) const
  {
    // J_smooth = sum || q_{i+1} - 2 q_i + q_{i-1} ||^2
    for (size_t i = 1; i + 1 < q.size(); ++i) {
      Vec3 a{q[i + 1].x - 2.0 * q[i].x + q[i - 1].x,
             q[i + 1].y - 2.0 * q[i].y + q[i - 1].y,
             q[i + 1].z - 2.0 * q[i].z + q[i - 1].z};
      grad[i - 1].x += lambda_smooth_ * 2.0 * a.x; grad[i - 1].y += lambda_smooth_ * 2.0 * a.y; grad[i - 1].z += lambda_smooth_ * 2.0 * a.z;
      grad[i].x     -= lambda_smooth_ * 4.0 * a.x; grad[i].y     -= lambda_smooth_ * 4.0 * a.y; grad[i].z     -= lambda_smooth_ * 4.0 * a.z;
      grad[i + 1].x += lambda_smooth_ * 2.0 * a.x; grad[i + 1].y += lambda_smooth_ * 2.0 * a.y; grad[i + 1].z += lambda_smooth_ * 2.0 * a.z;
    }
  }
  void addCollisionGradient(const std::vector<geometry_msgs::msg::Point> & q,
                            std::vector<Vec3> & grad) const
  {
    // J_collision = sum_{d<d_safe} (d_safe - d)^2 ,  d and grad(d) read from the ESDF
    for (size_t i = 0; i < q.size(); ++i) {
      Vec3 gd;
      const double d = esdfValueAndGradient(Vec3{q[i].x, q[i].y, q[i].z}, gd);
      if (d >= d_safe_) continue;
      const double coef = -2.0 * lambda_collision_ * (d_safe_ - d);
      grad[i].x += coef * gd.x; grad[i].y += coef * gd.y; grad[i].z += coef * gd.z;
    }
  }
  void addFeasibilityGradient(const std::vector<geometry_msgs::msg::Point> & q,
                              std::vector<Vec3> & grad) const
  {
    // Velocity control points V_i = (q_{i+1}-q_i)/dt ; penalise per-axis over-speed.
    const double ts = knot_dt_;
    auto axisOf = [](const geometry_msgs::msg::Point & p, int a) {
      return a == 0 ? p.x : (a == 1 ? p.y : p.z);
    };
    for (size_t i = 0; i + 1 < q.size(); ++i) {
      for (int a = 0; a < 3; ++a) {
        const double v = (axisOf(q[i + 1], a) - axisOf(q[i], a)) / ts;
        const double over = std::abs(v) - max_vel_;
        if (over <= 0.0) continue;
        const double g = 2.0 * lambda_feasible_ * over * (v > 0 ? 1.0 : -1.0) / ts;
        if (a == 0) { grad[i + 1].x += g; grad[i].x -= g; }
        else if (a == 1) { grad[i + 1].y += g; grad[i].y -= g; }
        else { grad[i + 1].z += g; grad[i].z -= g; }
      }
    }
    // Acceleration control points A_i = (q_{i+2}-2q_{i+1}+q_i)/dt^2.
    const double ts2 = ts * ts;
    for (size_t i = 0; i + 2 < q.size(); ++i) {
      for (int a = 0; a < 3; ++a) {
        const double acc = (axisOf(q[i + 2], a) - 2.0 * axisOf(q[i + 1], a) + axisOf(q[i], a)) / ts2;
        const double over = std::abs(acc) - max_acc_;
        if (over <= 0.0) continue;
        const double g = 2.0 * lambda_feasible_ * over * (acc > 0 ? 1.0 : -1.0) / ts2;
        if (a == 0) { grad[i].x += g; grad[i + 1].x -= 2.0 * g; grad[i + 2].x += g; }
        else if (a == 1) { grad[i].y += g; grad[i + 1].y -= 2.0 * g; grad[i + 2].y += g; }
        else { grad[i].z += g; grad[i + 1].z -= 2.0 * g; grad[i + 2].z += g; }
      }
    }
  }
  std::vector<geometry_msgs::msg::Point> optimizeControlPoints()
  {
    auto q = init_control_points_;
    for (int iter = 0; iter < opt_iterations_; ++iter) {
      std::vector<Vec3> grad(q.size(), Vec3{0.0, 0.0, 0.0});
      addSmoothnessGradient(q, grad);
      addCollisionGradient(q, grad);
      addFeasibilityGradient(q, grad);
      // The first/last two clamped control points are fixed (boundary conditions).
      // Each move is capped to max_move_ so the stiff feasibility term cannot blow up.
      for (size_t i = 2; i + 2 < q.size(); ++i) {
        Vec3 step{-opt_step_ * grad[i].x, -opt_step_ * grad[i].y, -opt_step_ * grad[i].z};
        const double n = norm(step);
        if (n > max_move_) { const double s = max_move_ / n; step.x *= s; step.y *= s; step.z *= s; }
        q[i].x += step.x; q[i].y += step.y; q[i].z += step.z;
        q[i].x = std::clamp(q[i].x, x_min_ + 0.2, x_max_ - 0.2);
        q[i].y = std::clamp(q[i].y, y_min_ + 0.2, y_max_ - 0.2);
        q[i].z = std::clamp(q[i].z, z_min_ + 0.2, z_max_ - 0.2);
      }
    }
    return q;
  }

  void runBsplineBackend()
  {
    generateControlPointsFromKinoPath();

    // ESDF construction (timed separately).
    const auto te0 = std::chrono::steady_clock::now();
    buildEsdf();
    const auto te1 = std::chrono::steady_clock::now();
    esdf_build_ms_ = std::chrono::duration<double, std::milli>(te1 - te0).count();

    // BEFORE: B-spline through the raw (un-optimised) control points.
    init_path_ = sampleBspline(init_control_points_);

    // Optimisation (timed).
    const auto to0 = std::chrono::steady_clock::now();
    optimized_control_points_ = optimizeControlPoints();
    const auto to1 = std::chrono::steady_clock::now();
    opt_ms_ = std::chrono::duration<double, std::milli>(to1 - to0).count();

    // AFTER: B-spline through the optimised control points.
    opt_path_ = sampleBspline(optimized_control_points_);

    // metrics: "before" = the raw front-end kinodynamic path (the trajectory prior
    // to B-spline optimisation); "after" = the optimised B-spline.
    init_smoothness_ = computeSmoothnessCost(kino_path_);
    opt_smoothness_ = computeSmoothnessCost(opt_path_);
    init_clearance_ = computeMinClearance(kino_path_);
    opt_clearance_ = computeMinClearance(opt_path_);
    init_length_ = computePathLength(kino_path_);
    opt_length_ = computePathLength(opt_path_);
    init_collision_free_ = checkCollisionFree(kino_path_);
    opt_collision_free_ = checkCollisionFree(opt_path_);
    double vi, ai, vo, ao;
    computeBsplineMaxVelAcc(init_control_points_, vi, ai);
    computeBsplineMaxVelAcc(optimized_control_points_, vo, ao);
    init_max_vel_ = vi; init_max_acc_ = ai; opt_max_vel_ = vo; opt_max_acc_ = ao;

    RCLCPP_INFO(this->get_logger(), "============ Fast-Planner ESDF back-end ============");
    RCLCPP_INFO(this->get_logger(),
      "ESDF grid: %d x %d x %d (%zu cells), build %.2f ms | optimise %.2f ms (%d iters)",
      esdf_nx_, esdf_ny_, esdf_nz_,
      static_cast<size_t>(esdf_nx_) * esdf_ny_ * esdf_nz_, esdf_build_ms_, opt_ms_, opt_iterations_);
    RCLCPP_INFO(this->get_logger(), "                        before   ->   after");
    RCLCPP_INFO(this->get_logger(), "smoothness cost      : %8.5f -> %8.5f", init_smoothness_, opt_smoothness_);
    RCLCPP_INFO(this->get_logger(), "min clearance (m)    : %8.3f -> %8.3f", init_clearance_, opt_clearance_);
    RCLCPP_INFO(this->get_logger(), "trajectory length (m): %8.3f -> %8.3f", init_length_, opt_length_);
    RCLCPP_INFO(this->get_logger(), "max B-spline vel(m/s): %8.3f -> %8.3f  (limit %.1f)", init_max_vel_, opt_max_vel_, max_vel_);
    RCLCPP_INFO(this->get_logger(), "max B-spline acc(m/s2): %7.3f -> %8.3f  (limit %.1f)", init_max_acc_, opt_max_acc_, max_acc_);
    RCLCPP_INFO(this->get_logger(), "collision-free       : %8s -> %8s",
      init_collision_free_ ? "true" : "false", opt_collision_free_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "====================================================");
  }

  // ------------------------------------------------------------------ metrics
  double computePathLength(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    double l = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
      const double dx = path[i].x - path[i - 1].x, dy = path[i].y - path[i - 1].y, dz = path[i].z - path[i - 1].z;
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
  // Max speed / acceleration from the B-spline derivative control points (exact bound).
  void computeBsplineMaxVelAcc(const std::vector<geometry_msgs::msg::Point> & q,
                               double & max_v, double & max_a) const
  {
    max_v = 0.0; max_a = 0.0;
    const double ts = knot_dt_;
    std::vector<Vec3> vel;
    for (size_t i = 0; i + 1 < q.size(); ++i) {
      Vec3 v{(q[i + 1].x - q[i].x) / ts, (q[i + 1].y - q[i].y) / ts, (q[i + 1].z - q[i].z) / ts};
      vel.push_back(v); max_v = std::max(max_v, norm(v));
    }
    for (size_t i = 0; i + 1 < vel.size(); ++i) {
      Vec3 a{(vel[i + 1].x - vel[i].x) / ts, (vel[i + 1].y - vel[i].y) / ts, (vel[i + 1].z - vel[i].z) / ts};
      max_a = std::max(max_a, norm(a));
    }
  }

  // ------------------------------------------------------------------ visualise
  void timerCallback()
  {
    visualization_msgs::msg::MarkerArray array;
    addObstacleMarkers(array);
    addStartGoalMarkers(array);
    addEsdfSliceMarker(array);
    addLineMarker(array, "kino_astar_path", 2000, kino_path_, 0.06, 0.1, 0.9, 0.1);
    addLineMarker(array, "init_bspline", 2500, init_path_, 0.05, 0.6, 0.6, 0.6);
    addLineMarker(array, "opt_bspline", 3000, opt_path_, 0.06, 1.0, 0.85, 0.1);
    addControlPointMarkers(array, optimized_control_points_);
    marker_pub_->publish(array);
  }
  void addObstacleMarkers(visualization_msgs::msg::MarkerArray & array)
  {
    int id = 0;
    for (const auto & obs : obstacles_) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map"; m.header.stamp = this->now();
      m.ns = "obstacles"; m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE; m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = obs.x; m.pose.position.y = obs.y; m.pose.position.z = obs.z; m.pose.orientation.w = 1.0;
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
      m.type = visualization_msgs::msg::Marker::SPHERE; m.action = visualization_msgs::msg::Marker::ADD;
      const Vec3 & p = (k == 0) ? start_pos_ : goal_pos_;
      m.pose.position.x = p.x; m.pose.position.y = p.y; m.pose.position.z = p.z; m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = 0.35;
      m.color.r = 0.0; m.color.g = (k == 0) ? 1.0 : 0.2; m.color.b = (k == 0) ? 0.0 : 1.0; m.color.a = 1.0;
      array.markers.push_back(m);
    }
  }
  // A horizontal ESDF slice at z = start height, coloured by distance (blue near -> red far).
  void addEsdfSliceMarker(visualization_msgs::msg::MarkerArray & array)
  {
    if (esdf_.empty()) return;
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = "esdf_slice"; m.id = 5000;
    m.type = visualization_msgs::msg::Marker::CUBE_LIST; m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = esdf_resolution_; m.scale.y = esdf_resolution_; m.scale.z = 0.02;
    m.pose.orientation.w = 1.0;
    const double z = start_pos_.z;
    const double d_max = 1.2;
    for (int ix = 0; ix < esdf_nx_; ix += 2) {
      for (int iy = 0; iy < esdf_ny_; iy += 2) {
        const double x = x_min_ + ix * esdf_resolution_;
        const double y = y_min_ + iy * esdf_resolution_;
        const double d = std::clamp(esdfValue(Vec3{x, y, z}), 0.0, d_max) / d_max;
        geometry_msgs::msg::Point p; p.x = x; p.y = y; p.z = 0.02;
        std_msgs::msg::ColorRGBA c; c.r = 1.0 - d; c.g = 0.15; c.b = d; c.a = 0.35;
        m.points.push_back(p); m.colors.push_back(c);
      }
    }
    array.markers.push_back(m);
  }
  void addLineMarker(visualization_msgs::msg::MarkerArray & array, const std::string & ns, int id,
                     const std::vector<geometry_msgs::msg::Point> & pts, double w, double r, double g, double b)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = ns; m.id = id; m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = w; m.pose.orientation.w = 1.0;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;
    m.points = pts;
    array.markers.push_back(m);
  }
  void addControlPointMarkers(visualization_msgs::msg::MarkerArray & array,
                              const std::vector<geometry_msgs::msg::Point> & cps)
  {
    int id = 4000;
    for (const auto & point : cps) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map"; m.header.stamp = this->now();
      m.ns = "control_points"; m.id = id++;
      m.type = visualization_msgs::msg::Marker::SPHERE; m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position = point; m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = 0.12;
      m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 0.95;
      array.markers.push_back(m);
    }
  }

  // ------------------------------------------------------------------ data
  std::string scenario_;

  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  double pos_resolution_, vel_resolution_;
  int nx_, ny_, nz_, nv_;
  double max_vel_, max_acc_, dt_;
  int primitive_check_num_;
  double goal_tolerance_, safety_margin_, max_search_time_;
  int max_expand_num_;
  Vec3 start_pos_, start_vel_, goal_pos_;

  int bspline_samples_per_segment_;
  double knot_dt_;
  double esdf_resolution_;
  int opt_iterations_;
  double opt_step_, max_move_, lambda_smooth_, lambda_collision_, lambda_feasible_, d_safe_;

  std::vector<BoxObstacle> obstacles_;
  std::vector<geometry_msgs::msg::Point> kino_path_;
  std::vector<geometry_msgs::msg::Point> init_control_points_;
  std::vector<geometry_msgs::msg::Point> optimized_control_points_;
  std::vector<geometry_msgs::msg::Point> init_path_;
  std::vector<geometry_msgs::msg::Point> opt_path_;

  std::vector<double> esdf_;
  int esdf_nx_ = 0, esdf_ny_ = 0, esdf_nz_ = 0;

  double frontend_ms_ = 0.0, esdf_build_ms_ = 0.0, opt_ms_ = 0.0, traj_duration_ = 0.0;
  double init_smoothness_ = 0.0, opt_smoothness_ = 0.0;
  double init_clearance_ = 0.0, opt_clearance_ = 0.0;
  double init_length_ = 0.0, opt_length_ = 0.0;
  double init_max_vel_ = 0.0, opt_max_vel_ = 0.0, init_max_acc_ = 0.0, opt_max_acc_ = 0.0;
  bool init_collision_free_ = false, opt_collision_free_ = false;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BsplineBackendDemoNode>());
  rclcpp::shutdown();
  return 0;
}
