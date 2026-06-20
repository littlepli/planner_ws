// closed_loop_demo.cpp
//
// Task 15-④: full "perception -> planning -> control" closed loop.
//
//   perception : a box-obstacle map is generated and treated as the sensed world.
//   planning   : Kinodynamic A* front-end produces an initial path, then the
//                EGO-Planner-style ESDF-free B-spline back-end optimises it into a
//                smooth, safe trajectory, which is time-parameterised.
//   control    : a PD + feed-forward tracking controller drives a double-integrator
//                robot model along the optimised trajectory.
//
// Everything is published to /planner_core_demo/closed_loop_markers for RViz and the
// robot state to /closed_loop_demo/odom.  The node prints the live tracking error
// and a final closed-loop report (completion time, RMS / max tracking error,
// whether the executed path stayed collision-free).

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
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

using namespace std::chrono_literals;

struct Vec3 { double x; double y; double z; };
struct BoxObstacle { double x; double y; double z; double sx; double sy; double sz; };
struct KinoNode { Vec3 pos; Vec3 vel; double time; double g_score; double f_score; int parent; long long key; };
struct QueueItem { double f_score; int node_id; bool operator<(const QueueItem & o) const { return f_score > o.f_score; } };
struct DesiredState { Vec3 pos; Vec3 vel; Vec3 acc; };

class ClosedLoopDemoNode : public rclcpp::Node
{
public:
  ClosedLoopDemoNode() : Node("closed_loop_demo_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/closed_loop_markers", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/closed_loop_demo/odom", 10);

    initParams();
    generateObstacles();          // perception
    assignObstacleVelocities();   // make a few obstacles move (dynamic world)

    if (planTrajectory()) {       // planning (front-end + back-end + time alloc)
      robot_pos_ = start_pos_;
      robot_vel_ = Vec3{0.0, 0.0, 0.0};
      planning_ok_ = true;
      RCLCPP_INFO(this->get_logger(),
        "Planning done: %zu trajectory points, duration %.2f s. Starting control loop.",
        ref_pos_.size(), total_time_);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Planning failed; control loop will idle.");
    }

    timer_ = this->create_wall_timer(20ms, std::bind(&ClosedLoopDemoNode::timerCallback, this));
    RCLCPP_INFO(this->get_logger(), "closed_loop_demo_node started.");
  }

private:
  // -------------------------------------------------------------- params
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
    nv_ = static_cast<int>(2.0 * max_vel_ / vel_resolution_) + 1;

    // Scenario type.
    scenario_ = this->declare_parameter<std::string>("scenario", "dense");

    opt_iterations_ = 300; opt_step_ = 0.06;
    lambda_smooth_ = 1.0; lambda_collision_ = 2.2; d_safe_ = 0.6;
    bspline_samples_per_segment_ = 12;
    cruise_speed_ = 1.6;

    start_pos_ = Vec3{-7.0, -6.0, 1.0}; start_vel_ = Vec3{0.0, 0.0, 0.0};
    goal_pos_ = Vec3{7.0, 6.0, 1.0};

    // Controller (double integrator, critically-damped-ish PD + feed-forward).
    ctrl_dt_ = 0.02; elapsed_time_ = 0.0;
    kp_ = 4.0; kd_ = 3.6; ctrl_max_acc_ = 6.0; ctrl_max_vel_ = 3.5;
    reached_goal_ = false; step_count_ = 0; planning_ok_ = false;
    err_sum_sq_ = 0.0; err_max_ = 0.0; err_count_ = 0; exec_collision_free_ = true;

    // ---- environment-upgrade options (configurable via ROS params) ----
    // dynamic obstacles + receding-horizon replanning
    dynamic_ = this->declare_parameter<bool>("dynamic", true);
    replan_period_ = this->declare_parameter<double>("replan_period", 0.25);
    replan_iterations_ = 120;          // warm-started EGO iterations per replan (cheap)
    last_replan_time_ = 0.0; replan_count_ = 0;
    // more realistic robot: control delay + process noise + steady disturbance
    disturbance_ = this->declare_parameter<bool>("disturbance", true);
    accel_noise_std_ = 0.35;           // m/s^2 Gaussian process noise on commanded accel
    disturbance_acc_ = Vec3{0.25, -0.15, 0.0};   // constant "wind"-like disturbance
    prev_cmd_acc_ = Vec3{0.0, 0.0, 0.0};         // one-step actuation delay
    exec_collision_steps_ = 0;
    noise_rng_.seed(7);
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

  // Give a few obstacles a horizontal velocity so the world is dynamic.  They are
  // placed near the nominal corridor so the robot must actively re-plan around them.
  void assignObstacleVelocities()
  {
    obs_vel_.assign(obstacles_.size(), Vec3{0.0, 0.0, 0.0});
    if (!dynamic_) return;
    // Two movers crossing the path diagonally.
    obstacles_.push_back(BoxObstacle{-2.0, 1.5, 1.0, 0.7, 0.7, 2.0});
    obs_vel_.push_back(Vec3{0.55, -0.45, 0.0});
    obstacles_.push_back(BoxObstacle{2.6, -1.8, 1.0, 0.7, 0.7, 2.0});
    obs_vel_.push_back(Vec3{-0.5, 0.55, 0.0});
    dynamic_begin_ = obstacles_.size() - 2;   // index of first dynamic obstacle
  }

  // Advance dynamic obstacles and bounce them inside the arena.
  void updateObstacles(double dt)
  {
    if (!dynamic_) return;
    for (size_t i = 0; i < obstacles_.size(); ++i) {
      if (std::abs(obs_vel_[i].x) < 1e-9 && std::abs(obs_vel_[i].y) < 1e-9) continue;
      obstacles_[i].x += obs_vel_[i].x * dt;
      obstacles_[i].y += obs_vel_[i].y * dt;
      if (obstacles_[i].x < -4.5 || obstacles_[i].x > 4.5) obs_vel_[i].x = -obs_vel_[i].x;
      if (obstacles_[i].y < -3.8 || obstacles_[i].y > 3.8) obs_vel_[i].y = -obs_vel_[i].y;
    }
  }

  // -------------------------------------------------------------- helpers
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

  // ============================ PLANNING =============================== //
  // Full plan: front-end Kinodynamic A* + EGO back-end (run once at the start, or
  // as a fallback when local replanning cannot keep the trajectory safe).
  bool planTrajectory()
  {
    if (!runKinoAStar()) return false;
    generateControlPointsFromKinoPath();
    current_cps_ = optimizeEgoCps(cps_, opt_iterations_);   // persist the control points
    opt_path_ = sampleBspline(current_cps_);
    if (opt_path_.size() < 2) return false;
    timeAllocate();
    return true;
  }

  // Receding-horizon local replan: warm-start the EGO back-end from the CURRENT
  // control points against the latest obstacle snapshot (a few ms), then keep the
  // robot's progress by re-anchoring the tracking time to the nearest sample.
  void recedingReplan()
  {
    if (current_cps_.size() < 4) return;
    current_cps_ = optimizeEgoCps(current_cps_, replan_iterations_);
    opt_path_ = sampleBspline(current_cps_);
    timeAllocate();
    // re-anchor elapsed_time_ to the reference sample closest to the robot
    double best = std::numeric_limits<double>::infinity();
    size_t best_i = 0;
    for (size_t i = 0; i < ref_pos_.size(); ++i) {
      const double d = dist(robot_pos_, toVec(ref_pos_[i]));
      if (d < best) { best = d; best_i = i; }
    }
    elapsed_time_ = ref_time_[best_i];
    replan_count_++;
  }

  // ----- front-end (Kinodynamic A*) -----
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
    return true;
  }

  // ----- back-end (EGO-style ESDF-free B-spline optimisation) -----
  void generateControlPointsFromKinoPath()
  {
    cps_.clear();
    if (kino_path_.size() < 2) return;
    // Kinodynamic A* only reaches within goal_tolerance of the goal, so clamp the
    // spline's terminal control points to the EXACT goal: the executed trajectory
    // then actually arrives at the goal instead of stopping short.
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
  Vec3 egoCollisionGradient(const geometry_msgs::msg::Point & pt) const
  {
    const Vec3 q{pt.x, pt.y, pt.z};
    double best = std::numeric_limits<double>::infinity();
    const BoxObstacle * bo = nullptr;
    for (const auto & obs : obstacles_) { const double c = computeClearanceToBox(q, obs); if (c < best) { best = c; bo = &obs; } }
    if (bo == nullptr || best >= d_safe_) return Vec3{0.0, 0.0, 0.0};
    const double hx = bo->sx / 2.0, hy = bo->sy / 2.0, hz = bo->sz / 2.0;
    const double cx = std::clamp(q.x, bo->x - hx, bo->x + hx);
    const double cy = std::clamp(q.y, bo->y - hy, bo->y + hy);
    const double cz = std::clamp(q.z, bo->z - hz, bo->z + hz);
    Vec3 v{q.x - cx, q.y - cy, q.z - cz};
    double n = norm(v);
    if (n < 1e-6) {
      const double dxf = hx - std::abs(q.x - bo->x), dyf = hy - std::abs(q.y - bo->y), dzf = hz - std::abs(q.z - bo->z);
      if (dxf <= dyf && dxf <= dzf) v = Vec3{(q.x >= bo->x) ? 1.0 : -1.0, 0, 0};
      else if (dyf <= dzf)         v = Vec3{0, (q.y >= bo->y) ? 1.0 : -1.0, 0};
      else                          v = Vec3{0, 0, (q.z >= bo->z) ? 1.0 : -1.0};
      n = 1.0;
    }
    v.x /= n; v.y /= n; v.z /= n;
    const double coef = -2.0 * lambda_collision_ * (d_safe_ - best);
    return Vec3{coef * v.x, coef * v.y, coef * v.z};
  }
  // Optimise the B-spline control points with the EGO (ESDF-free) collision
  // gradient, starting from `seed`, for `iters` iterations.  Returns control points.
  std::vector<geometry_msgs::msg::Point> optimizeEgoCps(
    const std::vector<geometry_msgs::msg::Point> & seed, int iters)
  {
    auto q = seed;
    for (int iter = 0; iter < iters; ++iter) {
      std::vector<Vec3> grad(q.size(), Vec3{0.0, 0.0, 0.0});
      for (size_t i = 1; i + 1 < q.size(); ++i) {
        Vec3 a{q[i + 1].x - 2.0 * q[i].x + q[i - 1].x,
               q[i + 1].y - 2.0 * q[i].y + q[i - 1].y,
               q[i + 1].z - 2.0 * q[i].z + q[i - 1].z};
        grad[i - 1].x += lambda_smooth_ * 2.0 * a.x; grad[i - 1].y += lambda_smooth_ * 2.0 * a.y; grad[i - 1].z += lambda_smooth_ * 2.0 * a.z;
        grad[i].x     -= lambda_smooth_ * 4.0 * a.x; grad[i].y     -= lambda_smooth_ * 4.0 * a.y; grad[i].z     -= lambda_smooth_ * 4.0 * a.z;
        grad[i + 1].x += lambda_smooth_ * 2.0 * a.x; grad[i + 1].y += lambda_smooth_ * 2.0 * a.y; grad[i + 1].z += lambda_smooth_ * 2.0 * a.z;
      }
      for (size_t i = 0; i < q.size(); ++i) {
        Vec3 g = egoCollisionGradient(q[i]);
        grad[i].x += g.x; grad[i].y += g.y; grad[i].z += g.z;
      }
      for (size_t i = 2; i + 2 < q.size(); ++i) {
        q[i].x -= opt_step_ * grad[i].x; q[i].y -= opt_step_ * grad[i].y; q[i].z -= opt_step_ * grad[i].z;
        q[i].x = std::clamp(q[i].x, x_min_ + 0.2, x_max_ - 0.2);
        q[i].y = std::clamp(q[i].y, y_min_ + 0.2, y_max_ - 0.2);
        q[i].z = std::clamp(q[i].z, z_min_ + 0.2, z_max_ - 0.2);
      }
    }
    return q;
  }

  // ----- time allocation: assign a timestamp to every spline sample -----
  void timeAllocate()
  {
    ref_pos_ = opt_path_;
    ref_time_.assign(ref_pos_.size(), 0.0);
    for (size_t i = 1; i < ref_pos_.size(); ++i) {
      const double dx = ref_pos_[i].x - ref_pos_[i - 1].x;
      const double dy = ref_pos_[i].y - ref_pos_[i - 1].y;
      const double dz = ref_pos_[i].z - ref_pos_[i - 1].z;
      const double seg = std::sqrt(dx * dx + dy * dy + dz * dz);
      ref_time_[i] = ref_time_[i - 1] + seg / cruise_speed_;
    }
    total_time_ = ref_time_.empty() ? 0.0 : ref_time_.back();
  }

  // Interpolate desired pos/vel/acc at time t from the time-stamped samples.
  DesiredState referenceAt(double t) const
  {
    DesiredState d{};
    const size_t n = ref_pos_.size();
    if (n == 0) return d;
    if (t <= 0.0) { d.pos = toVec(ref_pos_.front()); return d; }
    if (t >= total_time_) { d.pos = toVec(ref_pos_.back()); return d; }
    // locate segment [i-1, i] with ref_time_[i-1] <= t < ref_time_[i]
    size_t i = 1;
    while (i < n && ref_time_[i] < t) ++i;
    if (i >= n) i = n - 1;
    const double t0 = ref_time_[i - 1], t1 = ref_time_[i];
    const double seg_dt = std::max(t1 - t0, 1e-6);
    const double a = (t - t0) / seg_dt;
    d.pos = Vec3{
      ref_pos_[i - 1].x + a * (ref_pos_[i].x - ref_pos_[i - 1].x),
      ref_pos_[i - 1].y + a * (ref_pos_[i].y - ref_pos_[i - 1].y),
      ref_pos_[i - 1].z + a * (ref_pos_[i].z - ref_pos_[i - 1].z)};
    d.vel = Vec3{
      (ref_pos_[i].x - ref_pos_[i - 1].x) / seg_dt,
      (ref_pos_[i].y - ref_pos_[i - 1].y) / seg_dt,
      (ref_pos_[i].z - ref_pos_[i - 1].z) / seg_dt};
    // acceleration: difference of neighbouring segment velocities (feed-forward)
    if (i + 1 < n) {
      const double t2 = ref_time_[i + 1];
      const double seg_dt2 = std::max(t2 - t1, 1e-6);
      Vec3 v_next{
        (ref_pos_[i + 1].x - ref_pos_[i].x) / seg_dt2,
        (ref_pos_[i + 1].y - ref_pos_[i].y) / seg_dt2,
        (ref_pos_[i + 1].z - ref_pos_[i].z) / seg_dt2};
      d.acc = Vec3{(v_next.x - d.vel.x) / seg_dt, (v_next.y - d.vel.y) / seg_dt, (v_next.z - d.vel.z) / seg_dt};
    }
    return d;
  }
  Vec3 toVec(const geometry_msgs::msg::Point & p) const { return Vec3{p.x, p.y, p.z}; }

  // ============================ CONTROL =============================== //
  Vec3 clampNorm(const Vec3 & v, double m) const
  {
    const double n = norm(v);
    if (n <= m || n < 1e-9) return v;
    const double s = m / n; return Vec3{v.x * s, v.y * s, v.z * s};
  }
  void controlStep()
  {
    DesiredState d = referenceAt(elapsed_time_);
    Vec3 pe{d.pos.x - robot_pos_.x, d.pos.y - robot_pos_.y, d.pos.z - robot_pos_.z};
    Vec3 ve{d.vel.x - robot_vel_.x, d.vel.y - robot_vel_.y, d.vel.z - robot_vel_.z};
    Vec3 cmd{kp_ * pe.x + kd_ * ve.x + d.acc.x,
             kp_ * pe.y + kd_ * ve.y + d.acc.y,
             kp_ * pe.z + kd_ * ve.z + d.acc.z};
    cmd = clampNorm(cmd, ctrl_max_acc_);

    // Realistic actuation: one-step delay + Gaussian process noise + steady disturbance.
    Vec3 acc = prev_cmd_acc_;          // command applied with a one control-step lag
    prev_cmd_acc_ = cmd;
    if (disturbance_) {
      std::normal_distribution<double> noise(0.0, accel_noise_std_);
      acc.x += noise(noise_rng_) + disturbance_acc_.x;
      acc.y += noise(noise_rng_) + disturbance_acc_.y;
      acc.z += noise(noise_rng_) + disturbance_acc_.z;
    }

    robot_pos_.x += robot_vel_.x * ctrl_dt_ + 0.5 * acc.x * ctrl_dt_ * ctrl_dt_;
    robot_pos_.y += robot_vel_.y * ctrl_dt_ + 0.5 * acc.y * ctrl_dt_ * ctrl_dt_;
    robot_pos_.z += robot_vel_.z * ctrl_dt_ + 0.5 * acc.z * ctrl_dt_ * ctrl_dt_;
    robot_vel_.x += acc.x * ctrl_dt_; robot_vel_.y += acc.y * ctrl_dt_; robot_vel_.z += acc.z * ctrl_dt_;
    robot_vel_ = clampNorm(robot_vel_, ctrl_max_vel_);

    current_desired_ = d;
    current_error_ = dist(robot_pos_, d.pos);
    err_sum_sq_ += current_error_ * current_error_; err_count_++;
    err_max_ = std::max(err_max_, current_error_);
    if (isInsideObstacle(robot_pos_)) { exec_collision_free_ = false; exec_collision_steps_++; }

    geometry_msgs::msg::Point ap; ap.x = robot_pos_.x; ap.y = robot_pos_.y; ap.z = robot_pos_.z;
    actual_path_.push_back(ap);
    if (actual_path_.size() > 4000) actual_path_.erase(actual_path_.begin());

    if (!reached_goal_ && dist(robot_pos_, goal_pos_) < 0.25 && norm(robot_vel_) < 0.4) {
      reached_goal_ = true;
      const double rms = (err_count_ > 0) ? std::sqrt(err_sum_sq_ / err_count_) : 0.0;
      RCLCPP_INFO(this->get_logger(), "============ CLOSED-LOOP REPORT ============");
      RCLCPP_INFO(this->get_logger(), "mode: dynamic=%s, disturbance=%s",
        dynamic_ ? "on" : "off", disturbance_ ? "on" : "off");
      RCLCPP_INFO(this->get_logger(), "goal reached at t = %.2f s", elapsed_time_);
      RCLCPP_INFO(this->get_logger(), "receding-horizon replans : %d", replan_count_);
      RCLCPP_INFO(this->get_logger(), "RMS tracking error : %.4f m", rms);
      RCLCPP_INFO(this->get_logger(), "max tracking error : %.4f m", err_max_);
      RCLCPP_INFO(this->get_logger(), "executed path collision-free: %s (%d colliding steps)",
        exec_collision_free_ ? "true" : "false", exec_collision_steps_);
      RCLCPP_INFO(this->get_logger(), "===========================================");
    }
  }

  void publishOdometry()
  {
    nav_msgs::msg::Odometry odom;
    odom.header.frame_id = "map"; odom.header.stamp = this->now();
    odom.child_frame_id = "base_link";
    odom.pose.pose.position.x = robot_pos_.x; odom.pose.pose.position.y = robot_pos_.y; odom.pose.pose.position.z = robot_pos_.z;
    odom.pose.pose.orientation.w = 1.0;
    odom.twist.twist.linear.x = robot_vel_.x; odom.twist.twist.linear.y = robot_vel_.y; odom.twist.twist.linear.z = robot_vel_.z;
    odom_pub_->publish(odom);
  }

  // -------------------------------------------------------------- markers
  void timerCallback()
  {
    if (planning_ok_ && !reached_goal_) {
      // perception: advance the dynamic world
      updateObstacles(ctrl_dt_);
      // planning: receding-horizon local replan against the latest obstacle snapshot
      if (dynamic_ && wall_clock_ - last_replan_time_ >= replan_period_) {
        recedingReplan();
        last_replan_time_ = wall_clock_;
      }
      // control
      controlStep();
      elapsed_time_ += ctrl_dt_;
      wall_clock_ += ctrl_dt_;
    }
    publishOdometry();
    publishMarkers();
    if (++step_count_ % 50 == 0 && planning_ok_) {
      RCLCPP_INFO(this->get_logger(),
        "t: %.2f s | pos: [%.2f, %.2f, %.2f] | err: %.3f m | replans: %d | reached: %s",
        wall_clock_, robot_pos_.x, robot_pos_.y, robot_pos_.z, current_error_,
        replan_count_, reached_goal_ ? "true" : "false");
    }
  }
  geometry_msgs::msg::Point toPoint(const Vec3 & v) const
  { geometry_msgs::msg::Point p; p.x = v.x; p.y = v.y; p.z = v.z; return p; }
  visualization_msgs::msg::Marker sphere(const std::string & ns, int id, const Vec3 & p,
    double s, double r, double g, double b, double a)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = ns; m.id = id; m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position = toPoint(p); m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = s;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
    return m;
  }
  void addLine(visualization_msgs::msg::MarkerArray & arr, const std::string & ns, int id,
    const std::vector<geometry_msgs::msg::Point> & pts, double w, double r, double g, double b, double lift = 0.0)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map"; m.header.stamp = this->now();
    m.ns = ns; m.id = id; m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = w; m.pose.orientation.w = 1.0;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;
    for (auto p : pts) { p.z += lift; m.points.push_back(p); }
    arr.markers.push_back(m);
  }
  void publishMarkers()
  {
    visualization_msgs::msg::MarkerArray arr;
    int id = 0;
    for (size_t k = 0; k < obstacles_.size(); ++k) {
      const auto & obs = obstacles_[k];
      const bool is_dynamic = (k < obs_vel_.size()) &&
        (std::abs(obs_vel_[k].x) > 1e-9 || std::abs(obs_vel_[k].y) > 1e-9);
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map"; m.header.stamp = this->now();
      m.ns = "obstacles"; m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE; m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = obs.x; m.pose.position.y = obs.y; m.pose.position.z = obs.z; m.pose.orientation.w = 1.0;
      m.scale.x = obs.sx; m.scale.y = obs.sy; m.scale.z = obs.sz;
      // static obstacles red, moving obstacles orange (clearly distinguishable).
      m.color.r = is_dynamic ? 1.0 : 0.8; m.color.g = is_dynamic ? 0.55 : 0.1;
      m.color.b = 0.1; m.color.a = is_dynamic ? 0.85 : 0.7;
      arr.markers.push_back(m);
    }
    addLine(arr, "kino_path", 100, kino_path_, 0.05, 0.1, 0.85, 0.1);
    addLine(arr, "reference_path", 101, ref_pos_, 0.09, 1.0, 0.85, 0.05);
    addLine(arr, "actual_path", 102, actual_path_, 0.07, 0.0, 0.9, 1.0, 0.10);
    arr.markers.push_back(sphere("start_goal", 200, start_pos_, 0.35, 0.0, 1.0, 0.0, 1.0));
    arr.markers.push_back(sphere("start_goal", 201, goal_pos_, 0.35, 0.0, 0.2, 1.0, 1.0));
    arr.markers.push_back(sphere("robot", 300, robot_pos_, 0.32, 1.0, 0.55, 0.0, 1.0));
    arr.markers.push_back(sphere("desired_state", 301, current_desired_.pos, 0.20, 1.0, 0.0, 1.0, 1.0));
    marker_pub_->publish(arr);
  }

  // -------------------------------------------------------------- data
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  double pos_resolution_, vel_resolution_;
  int nx_, ny_, nz_, nv_;
  double max_vel_, max_acc_, dt_;
  int primitive_check_num_;
  double goal_tolerance_, safety_margin_, max_search_time_;
  int max_expand_num_;
  Vec3 start_pos_, start_vel_, goal_pos_;

  std::string scenario_;

  int opt_iterations_;
  double opt_step_, lambda_smooth_, lambda_collision_, d_safe_;
  int bspline_samples_per_segment_;
  double cruise_speed_;

  double ctrl_dt_, elapsed_time_, total_time_ = 0.0;
  double kp_, kd_, ctrl_max_acc_, ctrl_max_vel_;
  bool reached_goal_, planning_ok_;
  int step_count_;

  double err_sum_sq_, err_max_; int err_count_; bool exec_collision_free_;
  double current_error_ = 0.0;
  DesiredState current_desired_{};

  // dynamic world + receding-horizon replanning
  bool dynamic_ = true;
  double replan_period_ = 0.25, last_replan_time_ = 0.0, wall_clock_ = 0.0;
  int replan_iterations_ = 120, replan_count_ = 0;
  size_t dynamic_begin_ = 0;
  std::vector<Vec3> obs_vel_;
  // realistic actuation: delay + noise + disturbance
  bool disturbance_ = true;
  double accel_noise_std_ = 0.0;
  Vec3 disturbance_acc_{}, prev_cmd_acc_{};
  int exec_collision_steps_ = 0;
  std::mt19937 noise_rng_;

  Vec3 robot_pos_{}, robot_vel_{};

  std::vector<BoxObstacle> obstacles_;
  std::vector<geometry_msgs::msg::Point> kino_path_;
  std::vector<geometry_msgs::msg::Point> cps_;
  std::vector<geometry_msgs::msg::Point> current_cps_;
  std::vector<geometry_msgs::msg::Point> opt_path_;
  std::vector<geometry_msgs::msg::Point> ref_pos_;
  std::vector<double> ref_time_;
  std::vector<geometry_msgs::msg::Point> actual_path_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClosedLoopDemoNode>());
  rclcpp::shutdown();
  return 0;
}
