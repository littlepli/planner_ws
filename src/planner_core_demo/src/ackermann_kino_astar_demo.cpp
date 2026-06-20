// ackermann_kino_astar_demo.cpp
//
// Task 15-① Ackermann variant: Kinodynamic A* front-end for a car-like vehicle.
//
// State space  : (x, y, θ, v)  — 2D position, heading, scalar speed.
// Control      : (v_cmd, δ)    — discrete speed and steering-angle samples.
// Motion model : bicycle kinematics  ẋ=v·cosθ, ẏ=v·sinθ, θ̇=v·tanδ/L.
//
// The rest of the A* machinery (priority queue, state-key hashing, heuristic)
// follows the same pattern as the 3D point-mass version (kino_astar_demo.cpp).
//
// Ackermann vehicle parameters (see README §Ackermann for a rationale):
//   wheelbase L = 1.5 m
//   max steering δ_max = 35° (0.6109 rad)
//   cruising speed = 1.5 m/s  (optional half-speed 0.75 m/s)
//   max speed      = 2.0 m/s
//
// This node publishes the generated path to /planner_core_demo/markers.

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

struct AckermannState
{
  double x;      // world x [m]
  double y;      // world y [m]
  double theta;  // heading [rad]
  double v;      // scalar speed [m/s]
};

struct KinoNode
{
  AckermannState s;
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
  bool operator<(const QueueItem & o) const { return f_score > o.f_score; }
};

class AckermannKinoAStarDemoNode : public rclcpp::Node
{
public:
  AckermannKinoAStarDemoNode() : Node("ackermann_kino_astar_demo_node")
  {
    RCLCPP_INFO(this->get_logger(), "constructor start");
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/markers", 10);

    initParams();
    RCLCPP_INFO(this->get_logger(), "params done, scenario=%s, obstacles count will follow",
      scenario_.c_str());

    generateObstacles();
    RCLCPP_INFO(this->get_logger(), "obstacles generated: %zu", obstacles_.size());

    runKinoAStar();

    timer_ = this->create_wall_timer(
      500ms, std::bind(&AckermannKinoAStarDemoNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "ackermann_kino_astar_demo_node started.");
  }

private:
  // ---------------------------------------------------------------- params
  void initParams()
  {
    // Arena: 16 × 14 m, fixed height z0 = 1.0 m.
    x_min_ = -8.0;  x_max_ = 8.0;
    y_min_ = -7.0;  y_max_ = 7.0;
    z0_   = 1.0;

    // Discrete state resolution.
    pos_resolution_ = 0.4;
    theta_resolution_ = M_PI / 12.0;   // 15°
    vel_resolution_  = 0.5;

    // Grid dimensions.
    nx_ = static_cast<int>((x_max_ - x_min_) / pos_resolution_);
    ny_ = static_cast<int>((y_max_ - y_min_) / pos_resolution_);
    ntheta_ = static_cast<int>(2.0 * M_PI / theta_resolution_);

    // Vehicle limits.
    wheelbase_ = 1.5;
    max_vel_   = 2.0;
    max_steer_ = 35.0 * M_PI / 180.0;   // 0.6109 rad

    // Cruising speed used by the primitives.
    v_cruise_  = 1.5;

    dt_ = 0.4;
    primitive_check_num_ = 8;
    goal_tolerance_ = 0.65;
    goal_theta_tol_ = M_PI / 6.0;        // ±30° heading tolerance at goal
    safety_margin_ = 0.22;               // slightly larger for car footprint
    max_search_time_ = 25.0;
    max_expand_num_ = 200000;

    start_state_.x = -7.0; start_state_.y = -6.0;
    start_state_.theta = std::atan2(6.0, 7.0);   // roughly towards goal
    start_state_.v = 0.0;

    goal_x_ = 7.0; goal_y_ = 6.0;

    nv_ = static_cast<int>(2.0 * max_vel_ / vel_resolution_) + 1;

    // Scenario type (hardcoded default, can be overridden via --ros-args -p scenario:=...).
    scenario_ = this->declare_parameter("scenario", std::string("dense"));
  }

  void generateObstacles()
  {
    obstacles_.clear();
    const std::string scen = scenario_;
    std::mt19937 rng(12);

    auto tooClose = [&](double ox, double oy) {
      return std::hypot(ox - start_state_.x, oy - start_state_.y) < 2.0 ||
             std::hypot(ox - goal_x_, oy - goal_y_) < 2.0;
    };

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
        if (tooClose(obs.x, obs.y)) continue;
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
          if (tooClose(obs.x, obs.y)) continue;
          obstacles_.push_back(obs);
        }
      }
    } else {
      // "dense"
      std::uniform_real_distribution<double> pos_x(-7.2, 7.2), pos_y(-6.2, 6.2);
      std::uniform_real_distribution<double> size_xy(0.4, 0.85), height(0.9, 2.8);
      for (int i = 0; i < 40; ++i) {
        BoxObstacle obs;
        obs.x = pos_x(rng); obs.y = pos_y(rng);
        obs.sx = size_xy(rng); obs.sy = size_xy(rng);
        obs.sz = height(rng); obs.z = obs.sz / 2.0;
        if (tooClose(obs.x, obs.y)) continue;
        obstacles_.push_back(obs);
      }
    }
    obstacles_.push_back(BoxObstacle{-1.2, -0.5, 1.6, 1.0, 2.8, 3.2});
    obstacles_.push_back(BoxObstacle{ 1.5,  1.2, 1.5, 2.4, 1.0, 3.0});
    obstacles_.push_back(BoxObstacle{ 2.5, -1.8, 1.3, 1.0, 2.6, 2.6});
  }

  // ---------------------------------------------------------------- helpers
  double distance2D(double x1, double y1, double x2, double y2) const
  { return std::hypot(x1 - x2, y1 - y2); }
  double angleDiff(double a, double b) const
  {
    double d = a - b;
    while (d >  M_PI) d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return d;
  }

  bool isInsideObstacle(double x, double y) const
  {
    for (const auto & obs : obstacles_) {
      if (std::abs(x - obs.x) <= obs.sx / 2.0 + safety_margin_ &&
          std::abs(y - obs.y) <= obs.sy / 2.0 + safety_margin_) {
        return true;
      }
    }
    return false;
  }

  bool isStateValid(const AckermannState & s) const
  {
    if (s.x < x_min_ || s.x >= x_max_ || s.y < y_min_ || s.y >= y_max_)
      return false;
    if (isInsideObstacle(s.x, s.y))
      return false;
    if (s.v < -max_vel_ - 1e-6 || s.v > max_vel_ + 1e-6)
      return false;
    return true;
  }

  // State key: discretise (x, y, theta, v) into a single integer.
  long long stateKey(const AckermannState & s) const
  {
    int ix = static_cast<int>((s.x - x_min_) / pos_resolution_);
    int iy = static_cast<int>((s.y - y_min_) / pos_resolution_);
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_) return -1;

    // wrap theta into [0, 2π)
    double th = std::fmod(s.theta, 2.0 * M_PI);
    if (th < 0.0) th += 2.0 * M_PI;
    int ith = static_cast<int>(th / theta_resolution_);
    if (ith < 0 || ith >= ntheta_) return -1;

    int iv = static_cast<int>(std::round((s.v + max_vel_) / vel_resolution_));
    if (iv < 0 || iv >= nv_) return -1;

    long long key = ix;
    key = key * ny_ + iy;
    key = key * ntheta_ + ith;
    key = key * nv_ + iv;
    return key;
  }

  // Bicycle kinematics: propagate state forward by dt.
  AckermannState propagate(const AckermannState & s, double v_cmd, double delta, double dt) const
  {
    // Clamp commanded speed and steering.
    double v = std::clamp(v_cmd, -max_vel_, max_vel_);
    double d = std::clamp(delta, -max_steer_, max_steer_);

    // Bicycle model.
    double dtheta = v * std::tan(d) / wheelbase_ * dt;
    double theta_mid = s.theta + 0.5 * dtheta;
    AckermannState ns;
    ns.x = s.x + v * std::cos(theta_mid) * dt;
    ns.y = s.y + v * std::sin(theta_mid) * dt;
    ns.theta = std::fmod(s.theta + dtheta, 2.0 * M_PI);
    if (ns.theta < 0.0) ns.theta += 2.0 * M_PI;
    ns.v = v;
    return ns;
  }

  // Sample discrete control pairs (v_cmd, delta).
  struct ControlSample { double v; double delta; };
  std::vector<ControlSample> generateControlSamples() const
  {
    std::vector<ControlSample> cs;
    // Forward speed options.
    for (double v : {0.5 * v_cruise_, v_cruise_}) {
      for (double s : {-max_steer_, -0.5 * max_steer_, 0.0, 0.5 * max_steer_, max_steer_}) {
        cs.push_back({v, s});
      }
    }
    // One reverse option to allow manoeuvring (three-point turns).
    cs.push_back({-0.4 * v_cruise_, 0.0});
    return cs;
  }

  bool checkPrimitiveCollision(const AckermannState & s0, double v_cmd, double delta) const
  {
    for (int i = 1; i <= primitive_check_num_; ++i) {
      const double t = dt_ * static_cast<double>(i) / primitive_check_num_;
      AckermannState si = propagate(s0, v_cmd, delta, t);
      if (!isStateValid(si)) return false;
    }
    return true;
  }

  double heuristic(const AckermannState & s) const
  {
    const double d = std::hypot(s.x - goal_x_, s.y - goal_y_);
    const double time_lb = d / max_vel_;

    // Heading alignment term — prefer pointing toward the goal.
    double desired_theta = std::atan2(goal_y_ - s.y, goal_x_ - s.x);
    double heading_err = std::abs(angleDiff(s.theta, desired_theta));
    const double heading_penalty = 0.15 * heading_err;

    return time_lb + heading_penalty;
  }

  // ================================================================ search
  bool runKinoAStar()
  {
    RCLCPP_INFO(this->get_logger(), "search start");
    const auto t0 = std::chrono::steady_clock::now();

    if (!isStateValid(start_state_)) {
      RCLCPP_ERROR(this->get_logger(), "Start state invalid.");
      return false;
    }
    RCLCPP_INFO(this->get_logger(), "start state valid");

    const auto controls = generateControlSamples();
    RCLCPP_INFO(this->get_logger(), "controls: %zu samples", controls.size());

    std::vector<KinoNode> nodes;
    std::priority_queue<QueueItem> open_set;
    std::unordered_map<long long, double> best_g;
    std::unordered_map<long long, int> best_id;
    std::unordered_set<long long> closed;

    long long sk = stateKey(start_state_);
    KinoNode start_node;
    start_node.s = start_state_;
    start_node.time = 0.0;
    start_node.g_score = 0.0;
    start_node.f_score = heuristic(start_state_);
    start_node.parent = -1;
    start_node.key = sk;
    nodes.push_back(start_node);
    best_g[sk] = 0.0;
    best_id[sk] = 0;
    open_set.push(QueueItem{start_node.f_score, 0});

    int final_id = -1, expand = 0;
    RCLCPP_INFO(this->get_logger(), "entering search loop");

    while (!open_set.empty() && expand < max_expand_num_) {
      QueueItem item = open_set.top(); open_set.pop();
      int cur_id = item.node_id;
      const KinoNode cur = nodes[cur_id];

      auto bit = best_id.find(cur.key);
      if (bit == best_id.end() || bit->second != cur_id) continue;
      if (closed.count(cur.key)) continue;
      closed.insert(cur.key);
      expand++;

      // Goal check: close to the target and heading roughly correct.
      if (std::hypot(cur.s.x - goal_x_, cur.s.y - goal_y_) < goal_tolerance_ &&
          cur.time > 0.5 &&
          std::abs(angleDiff(cur.s.theta,
                             std::atan2(goal_y_ - cur.s.y, goal_x_ - cur.s.x))) < goal_theta_tol_) {
        final_id = cur_id; break;
      }
      if (cur.time > max_search_time_) continue;

      for (const auto & c : controls) {
        if (!checkPrimitiveCollision(cur.s, c.v, c.delta)) continue;
        AckermannState ns = propagate(cur.s, c.v, c.delta, dt_);
        long long nk = stateKey(ns);
        if (nk < 0 || closed.count(nk)) continue;

        // Cost: time step + small penalty for steering (encourage straight paths)
        const double steer_penalty = 0.02 * std::abs(c.delta) * dt_;
        const double ng = cur.g_score + dt_ + steer_penalty;

        auto git = best_g.find(nk);
        if (git == best_g.end() || ng < git->second) {
          KinoNode nn;
          nn.s = ns; nn.time = cur.time + dt_;
          nn.g_score = ng; nn.f_score = ng + heuristic(ns);
          nn.parent = cur_id; nn.key = nk;
          int nid = static_cast<int>(nodes.size());
          nodes.push_back(nn);
          best_g[nk] = ng; best_id[nk] = nid;
          open_set.push(QueueItem{nn.f_score, nid});
        }
      }
    }

    path_.clear();
    if (final_id < 0) {
      const auto t1 = std::chrono::steady_clock::now();
      planning_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
      expanded_nodes_ = expand;
      RCLCPP_ERROR(this->get_logger(),
        "Ackermann Kinodynamic A* failed. Expanded: %d, time: %.1f ms",
        expand, planning_time_ms_);
      return false;
    }

    // Reconstruct path.
    std::vector<int> ids;
    int id = final_id;
    while (id >= 0) { ids.push_back(id); id = nodes[id].parent; }
    std::reverse(ids.begin(), ids.end());
    for (int nid : ids) {
      geometry_msgs::msg::Point p;
      p.x = nodes[nid].s.x; p.y = nodes[nid].s.y; p.z = z0_;
      path_.push_back(p);
    }

    const auto t1 = std::chrono::steady_clock::now();
    planning_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    expanded_nodes_ = expand;
    path_length_ = computePathLength(path_);
    collision_free_ = checkPathCollisionFree();

    RCLCPP_INFO(this->get_logger(),
      "Ackermann Kino A* success. Expanded: %d, path points: %zu",
      expand, path_.size());
    RCLCPP_INFO(this->get_logger(),
      "Metrics | time_ms: %.1f | length_m: %.3f | duration_s: %.2f | free: %s",
      planning_time_ms_, path_length_, nodes[final_id].time,
      collision_free_ ? "true" : "false");
    return true;
  }

  // ---------------------------------------------------------------- metrics
  double computePathLength(const std::vector<geometry_msgs::msg::Point> & p) const
  {
    double l = 0.0;
    for (size_t i = 1; i < p.size(); ++i)
      l += std::hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
    return l;
  }

  bool checkPathCollisionFree() const
  {
    for (const auto & pt : path_) {
      if (pt.x < x_min_ || pt.x >= x_max_ || pt.y < y_min_ || pt.y >= y_max_)
        return false;
      if (isInsideObstacle(pt.x, pt.y)) return false;
    }
    return true;
  }

  // ---------------------------------------------------------------- markers
  void timerCallback()
  {
    visualization_msgs::msg::MarkerArray arr;

    // Obstacles.
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
      m.color.r = 0.8; m.color.g = 0.1; m.color.b = 0.1; m.color.a = 0.75;
      arr.markers.push_back(m);
    }

    // Start (green sphere).
    visualization_msgs::msg::Marker start;
    start.header.frame_id = "map"; start.header.stamp = this->now();
    start.ns = "start_goal"; start.id = 1000;
    start.type = visualization_msgs::msg::Marker::SPHERE;
    start.action = visualization_msgs::msg::Marker::ADD;
    start.pose.position.x = start_state_.x;
    start.pose.position.y = start_state_.y;
    start.pose.position.z = z0_;
    start.pose.orientation.w = 1.0;
    start.scale.x = 0.35; start.scale.y = 0.35; start.scale.z = 0.35;
    start.color.r = 0.0; start.color.g = 1.0; start.color.b = 0.0; start.color.a = 1.0;
    arr.markers.push_back(start);

    // Goal (blue sphere).
    visualization_msgs::msg::Marker goal = start;
    goal.id = 1001;
    goal.pose.position.x = goal_x_;
    goal.pose.position.y = goal_y_;
    goal.color.r = 0.0; goal.color.g = 0.2; goal.color.b = 1.0;
    arr.markers.push_back(goal);

    // Path line.
    visualization_msgs::msg::Marker path;
    path.header.frame_id = "map"; path.header.stamp = this->now();
    path.ns = "ackermann_path"; path.id = 2000;
    path.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path.action = visualization_msgs::msg::Marker::ADD;
    path.scale.x = 0.08;
    path.color.r = 0.1; path.color.g = 0.9; path.color.b = 0.1; path.color.a = 1.0;
    path.points = path_;
    arr.markers.push_back(path);

    marker_pub_->publish(arr);
  }

  // ---------------------------------------------------------------- data
  double x_min_, x_max_, y_min_, y_max_, z0_;
  double pos_resolution_, theta_resolution_, vel_resolution_;
  int nx_, ny_, ntheta_, nv_;
  double wheelbase_, max_vel_, max_steer_, v_cruise_, dt_;
  int primitive_check_num_;
  double goal_tolerance_, goal_theta_tol_, safety_margin_, max_search_time_;
  int max_expand_num_;
  AckermannState start_state_;
  double goal_x_, goal_y_;
  std::string scenario_;

  std::vector<BoxObstacle> obstacles_;
  std::vector<geometry_msgs::msg::Point> path_;

  double planning_time_ms_ = 0.0;
  int expanded_nodes_ = 0;
  double path_length_ = 0.0;
  bool collision_free_ = false;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AckermannKinoAStarDemoNode>());
  rclcpp::shutdown();
  return 0;
}