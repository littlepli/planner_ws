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

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"

using namespace std::chrono_literals;

struct Vec3
{
  double x;
  double y;
  double z;
};

struct BoxObstacle
{
  double x;
  double y;
  double z;
  double sx;
  double sy;
  double sz;
};

struct KinoNode
{
  Vec3 pos;
  Vec3 vel;
  Vec3 acc;
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

  bool operator<(const QueueItem & other) const
  {
    return f_score > other.f_score;
  }
};

class KinoAStarDemoNode : public rclcpp::Node
{
public:
  KinoAStarDemoNode() : Node("kino_astar_demo_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/markers", 10);

    initParams();
    generateObstacles();
    runKinoAStar();

    timer_ = this->create_wall_timer(
      500ms, std::bind(&KinoAStarDemoNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "kino_astar_demo_node started.");
  }

private:
  void initParams()
  {
    // Expanded arena: 16 x 14 x 4 m (was 12 x 10 x 3.25).
    x_min_ = -8.0;
    x_max_ = 8.0;
    y_min_ = -7.0;
    y_max_ = 7.0;
    z_min_ = 0.25;
    z_max_ = 4.25;

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

    // Choose scenario type:
    //   "dense"    – heavy random coverage forces the planner into narrow passages
    //   "clustered" – obstacles grouped, leaving wider channels between clusters
    //   "narrow"   – explicit wall corridors (best for comparing front-end vs back-end)
    scenario_ = this->declare_parameter<std::string>("scenario", "dense");

    nv_ = static_cast<int>(2.0 * max_vel_ / vel_resolution_) + 1;
  }

  void generateObstacles()
  {
    obstacles_.clear();
    const std::string scen = scenario_;
    std::mt19937 rng(12);

    if (scen == "narrow") {
      // Parallel wall corridors with a gap in the middle forcing a detour.
      // Three horizontal walls stacked vertically, each with one gap at a
      // different x-offset so the planner must zigzag.
      const double wall_z = 1.5, wall_h = 3.0;
      const double wall_span = 5.5;   // half-width of wall
      const double wall_thick = 0.35;
      const double gap_half = 1.2;    // half-width of the passable gap

      auto addWall = [&](double cy, double gx) {
        obstacles_.push_back(BoxObstacle{-wall_span - 0.1, cy, wall_z,
                                          wall_span - gap_half + 0.1, wall_thick, wall_h});
        obstacles_.push_back(BoxObstacle{ gx + gap_half, cy, wall_z,
                                          wall_span - gap_half + 0.2, wall_thick, wall_h});
      };

      addWall(-1.5, -2.0);
      addWall( 0.0,  1.5);
      addWall( 1.3, -1.0);

      // Add scattered pillars to fill the space.
      std::uniform_real_distribution<double> px(-6.5, 6.5);
      std::uniform_real_distribution<double> py(-5.8, 5.8);
      std::uniform_real_distribution<double> sz(0.4, 0.7);
      std::uniform_real_distribution<double> h(0.8, 2.2);
      for (int i = 0; i < 35; ++i) {
        BoxObstacle obs;
        obs.x = px(rng); obs.y = py(rng);
        obs.sx = sz(rng); obs.sy = sz(rng);
        obs.sz = h(rng); obs.z = obs.sz / 2.0;
        if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
        if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
        bool blocked = false;
        for (const auto & e : obstacles_)  // don't overlap wall segments
          if (std::abs(obs.x - e.x) < e.sx / 2.0 + obs.sx / 2.0 + 0.3 &&
              std::abs(obs.y - e.y) < e.sy / 2.0 + obs.sy / 2.0 + 0.3) { blocked = true; break; }
        if (!blocked) obstacles_.push_back(obs);
      }
    } else if (scen == "clustered") {
      // Three obstacle clusters separated by wide channels.
      struct { double cx; double cy; int n; } clusters[] = {
        {-2.0, -2.0, 18}, {2.0, 1.5, 20}, {-0.5, 3.5, 16}
      };
      for (const auto & cl : clusters) {
        std::normal_distribution<double> cx(cl.cx, 2.0);
        std::normal_distribution<double> cy(cl.cy, 1.8);
        std::uniform_real_distribution<double> sz(0.5, 0.95);
        std::uniform_real_distribution<double> h(0.9, 2.6);
        for (int i = 0; i < cl.n; ++i) {
          BoxObstacle obs;
          obs.x = cx(rng); obs.y = cy(rng);
          obs.sx = sz(rng); obs.sy = sz(rng);
          obs.sz = h(rng); obs.z = obs.sz / 2.0;
          if (std::abs(obs.x) > 7.5 || std::abs(obs.y) > 6.5) continue;
          if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
          if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
          obstacles_.push_back(obs);
        }
      }
    } else {
      // "dense" (default) – heavy random coverage, roughly 60 boxes.
      std::uniform_real_distribution<double> pos_x(-7.2, 7.2);
      std::uniform_real_distribution<double> pos_y(-6.2, 6.2);
      std::uniform_real_distribution<double> size_xy(0.4, 0.85);
      std::uniform_real_distribution<double> height(0.9, 2.8);
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

    // A few large wall-like obstacles in the centre to force a clear detour.
    obstacles_.push_back(BoxObstacle{-1.2, -0.5, 1.6, 1.0, 2.8, 3.2});
    obstacles_.push_back(BoxObstacle{ 1.5,  1.2, 1.5, 2.4, 1.0, 3.0});
    obstacles_.push_back(BoxObstacle{ 2.5, -1.8, 1.3, 1.0, 2.6, 2.6});
  }

  double distance2D(double x1, double y1, double x2, double y2) const
  {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
  }

  double norm(const Vec3 & v) const
  {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  double dist(const Vec3 & a, const Vec3 & b) const
  {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  bool isInsideMap(const Vec3 & p) const
  {
    return p.x >= x_min_ && p.x < x_max_ &&
           p.y >= y_min_ && p.y < y_max_ &&
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
    if (!isInsideMap(p)) {
      return false;
    }

    if (isInsideObstacle(p)) {
      return false;
    }

    if (norm(v) > max_vel_ + 1e-6) {
      return false;
    }

    return true;
  }

  long long stateKey(const Vec3 & p, const Vec3 & v) const
  {
    int ix = static_cast<int>((p.x - x_min_) / pos_resolution_);
    int iy = static_cast<int>((p.y - y_min_) / pos_resolution_);
    int iz = static_cast<int>((p.z - z_min_) / pos_resolution_);

    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) {
      return -1;
    }

    int ivx = static_cast<int>(std::round((v.x + max_vel_) / vel_resolution_));
    int ivy = static_cast<int>(std::round((v.y + max_vel_) / vel_resolution_));
    int ivz = static_cast<int>(std::round((v.z + max_vel_) / vel_resolution_));

    if (ivx < 0 || ivx >= nv_ || ivy < 0 || ivy >= nv_ || ivz < 0 || ivz >= nv_) {
      return -1;
    }

    long long key = ix;
    key = key * ny_ + iy;
    key = key * nz_ + iz;
    key = key * nv_ + ivx;
    key = key * nv_ + ivy;
    key = key * nv_ + ivz;

    return key;
  }

  Vec3 propagatePos(const Vec3 & p, const Vec3 & v, const Vec3 & a, double t) const
  {
    return Vec3{
      p.x + v.x * t + 0.5 * a.x * t * t,
      p.y + v.y * t + 0.5 * a.y * t * t,
      p.z + v.z * t + 0.5 * a.z * t * t
    };
  }

  Vec3 propagateVel(const Vec3 & v, const Vec3 & a, double t) const
  {
    return Vec3{
      v.x + a.x * t,
      v.y + a.y * t,
      v.z + a.z * t
    };
  }

  bool checkPrimitiveCollision(const Vec3 & p, const Vec3 & v, const Vec3 & a) const
  {
    for (int i = 1; i <= primitive_check_num_; ++i) {
      const double t = dt_ * static_cast<double>(i) / static_cast<double>(primitive_check_num_);
      Vec3 pt = propagatePos(p, v, a, t);
      Vec3 vt = propagateVel(v, a, t);

      if (!isStateValid(pt, vt)) {
        return false;
      }
    }

    return true;
  }

  std::vector<Vec3> generateControlSamples() const
  {
    std::vector<Vec3> controls;
    const std::array<double, 3> values = {-max_acc_, 0.0, max_acc_};

    for (double ax : values) {
      for (double ay : values) {
        for (double az : values) {
          Vec3 a{ax, ay, az};
          if (norm(a) <= max_acc_ * 1.75) {
            controls.push_back(a);
          }
        }
      }
    }

    return controls;
  }

  double heuristic(const Vec3 & p, const Vec3 & v) const
  {
    const double d = dist(p, goal_pos_);
    const double time_lb = d / max_vel_;

    Vec3 to_goal{
      goal_pos_.x - p.x,
      goal_pos_.y - p.y,
      goal_pos_.z - p.z
    };

    const double d_norm = std::max(d, 1e-3);
    Vec3 desired_v{
      max_vel_ * to_goal.x / d_norm,
      max_vel_ * to_goal.y / d_norm,
      max_vel_ * to_goal.z / d_norm
    };

    const double vel_penalty = 0.08 * dist(v, desired_v);

    return time_lb + vel_penalty;
  }

  bool runKinoAStar()
  {
    const auto time_begin = std::chrono::steady_clock::now();

    if (!isStateValid(start_pos_, start_vel_)) {
      const auto time_end = std::chrono::steady_clock::now();
      planning_time_ms_ =
        std::chrono::duration<double, std::milli>(time_end - time_begin).count();

      RCLCPP_ERROR(
        this->get_logger(),
        "Start state is invalid. planning_time_ms: %.3f",
        planning_time_ms_);

      return false;
    }

    const auto controls = generateControlSamples();

    std::vector<KinoNode> nodes;
    std::priority_queue<QueueItem> open_set;
    std::unordered_map<long long, double> best_g;
    std::unordered_map<long long, int> best_id;
    std::unordered_set<long long> closed;

    long long start_key = stateKey(start_pos_, start_vel_);

    KinoNode start_node;
    start_node.pos = start_pos_;
    start_node.vel = start_vel_;
    start_node.acc = Vec3{0.0, 0.0, 0.0};
    start_node.time = 0.0;
    start_node.g_score = 0.0;
    start_node.f_score = heuristic(start_pos_, start_vel_);
    start_node.parent = -1;
    start_node.key = start_key;

    nodes.push_back(start_node);
    best_g[start_key] = 0.0;
    best_id[start_key] = 0;
    open_set.push(QueueItem{start_node.f_score, 0});

    int final_id = -1;
    int expand_count = 0;

    while (!open_set.empty() && expand_count < max_expand_num_) {
      QueueItem item = open_set.top();
      open_set.pop();

      int current_id = item.node_id;
      const KinoNode current = nodes[current_id];

      auto best_it = best_id.find(current.key);
      if (best_it == best_id.end() || best_it->second != current_id) {
        continue;
      }

      if (closed.find(current.key) != closed.end()) {
        continue;
      }

      closed.insert(current.key);
      expand_count++;

      if (dist(current.pos, goal_pos_) < goal_tolerance_ && current.time > 1.0) {
        final_id = current_id;
        break;
      }

      if (current.time > max_search_time_) {
        continue;
      }

      for (const auto & acc : controls) {
        if (!checkPrimitiveCollision(current.pos, current.vel, acc)) {
          continue;
        }

        Vec3 next_pos = propagatePos(current.pos, current.vel, acc, dt_);
        Vec3 next_vel = propagateVel(current.vel, acc, dt_);

        long long next_key = stateKey(next_pos, next_vel);
        if (next_key < 0) {
          continue;
        }

        if (closed.find(next_key) != closed.end()) {
          continue;
        }

        const double acc_cost = 0.03 * norm(acc) * norm(acc) * dt_;
        const double next_g = current.g_score + dt_ + acc_cost;

        auto g_it = best_g.find(next_key);
        if (g_it == best_g.end() || next_g < g_it->second) {
          KinoNode next_node;
          next_node.pos = next_pos;
          next_node.vel = next_vel;
          next_node.acc = acc;
          next_node.time = current.time + dt_;
          next_node.g_score = next_g;
          next_node.f_score = next_g + heuristic(next_pos, next_vel);
          next_node.parent = current_id;
          next_node.key = next_key;

          int new_id = static_cast<int>(nodes.size());
          nodes.push_back(next_node);

          best_g[next_key] = next_g;
          best_id[next_key] = new_id;
          open_set.push(QueueItem{next_node.f_score, new_id});
        }
      }
    }

    kino_path_.clear();
    kino_velocities_.clear();
    kino_accelerations_.clear();

    if (final_id < 0) {
      const auto time_end = std::chrono::steady_clock::now();
      planning_time_ms_ =
        std::chrono::duration<double, std::milli>(time_end - time_begin).count();
      expanded_nodes_ = expand_count;
      collision_free_ = false;

      RCLCPP_ERROR(
        this->get_logger(),
        "Kinodynamic A* failed. Expanded nodes: %d, planning_time_ms: %.3f",
        expanded_nodes_,
        planning_time_ms_);

      return false;
    }

    std::vector<int> node_ids;
    int id = final_id;

    while (id >= 0) {
      node_ids.push_back(id);
      id = nodes[id].parent;
    }

    std::reverse(node_ids.begin(), node_ids.end());

    for (const auto node_id : node_ids) {
      geometry_msgs::msg::Point p;
      p.x = nodes[node_id].pos.x;
      p.y = nodes[node_id].pos.y;
      p.z = nodes[node_id].pos.z;
      kino_path_.push_back(p);

      kino_velocities_.push_back(nodes[node_id].vel);
      kino_accelerations_.push_back(nodes[node_id].acc);
    }

    const auto time_end = std::chrono::steady_clock::now();
    planning_time_ms_ =
      std::chrono::duration<double, std::milli>(time_end - time_begin).count();

    expanded_nodes_ = expand_count;
    traj_length_ = computePathLength(kino_path_);
    traj_duration_ = nodes[final_id].time;
    min_clearance_ = computeMinClearance();
    max_velocity_ = computeMaxVelocity();
    avg_velocity_ = computeAverageVelocity();
    max_acceleration_ = computeMaxAcceleration();
    collision_free_ = checkPathCollisionFree();

    RCLCPP_INFO(
      this->get_logger(),
      "Kinodynamic A* success. Expanded nodes: %d, path points: %zu",
      expanded_nodes_,
      kino_path_.size());

    RCLCPP_INFO(
      this->get_logger(),
      "Metrics | planning_time_ms: %.3f | length_m: %.3f | duration_s: %.3f | min_clearance_m: %.3f",
      planning_time_ms_,
      traj_length_,
      traj_duration_,
      min_clearance_);

    RCLCPP_INFO(
      this->get_logger(),
      "Metrics | max_velocity_mps: %.3f | avg_velocity_mps: %.3f | max_acceleration_mps2: %.3f | collision_free: %s",
      max_velocity_,
      avg_velocity_,
      max_acceleration_,
      collision_free_ ? "true" : "false");

    return true;
  }

  double computeClearanceToBox(const Vec3 & p, const BoxObstacle & obs) const
  {
    const double dx = std::max(std::abs(p.x - obs.x) - obs.sx / 2.0, 0.0);
    const double dy = std::max(std::abs(p.y - obs.y) - obs.sy / 2.0, 0.0);
    const double dz = std::max(std::abs(p.z - obs.z) - obs.sz / 2.0, 0.0);

    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  double computeMinClearance() const
  {
    if (kino_path_.empty() || obstacles_.empty()) {
      return 0.0;
    }

    double min_clearance = std::numeric_limits<double>::infinity();

    for (const auto & point : kino_path_) {
      const Vec3 p{point.x, point.y, point.z};

      for (const auto & obs : obstacles_) {
        min_clearance = std::min(min_clearance, computeClearanceToBox(p, obs));
      }
    }

    if (!std::isfinite(min_clearance)) {
      return 0.0;
    }

    return min_clearance;
  }

  double computeMaxVelocity() const
  {
    double max_v = 0.0;

    for (const auto & v : kino_velocities_) {
      max_v = std::max(max_v, norm(v));
    }

    return max_v;
  }

  double computeAverageVelocity() const
  {
    if (kino_velocities_.empty()) {
      return 0.0;
    }

    double sum_v = 0.0;

    for (const auto & v : kino_velocities_) {
      sum_v += norm(v);
    }

    return sum_v / static_cast<double>(kino_velocities_.size());
  }

  double computeMaxAcceleration() const
  {
    double max_a = 0.0;

    for (const auto & a : kino_accelerations_) {
      max_a = std::max(max_a, norm(a));
    }

    return max_a;
  }

  bool checkPathCollisionFree() const
  {
    for (const auto & point : kino_path_) {
      const Vec3 p{point.x, point.y, point.z};

      if (!isInsideMap(p) || isInsideObstacle(p)) {
        return false;
      }
    }

    return true;
  }

  double computePathLength(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    double length = 0.0;

    for (size_t i = 1; i < path.size(); ++i) {
      double dx = path[i].x - path[i - 1].x;
      double dy = path[i].y - path[i - 1].y;
      double dz = path[i].z - path[i - 1].z;
      length += std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    return length;
  }

  void timerCallback()
  {
    visualization_msgs::msg::MarkerArray array;

    addObstacleMarkers(array);
    addStartGoalMarkers(array);
    addPathMarker(array);

    marker_pub_->publish(array);
  }

  void addObstacleMarkers(visualization_msgs::msg::MarkerArray & array)
  {
    int id = 0;

    for (const auto & obs : obstacles_) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map";
      m.header.stamp = this->now();
      m.ns = "obstacles";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;

      m.pose.position.x = obs.x;
      m.pose.position.y = obs.y;
      m.pose.position.z = obs.z;

      m.scale.x = obs.sx;
      m.scale.y = obs.sy;
      m.scale.z = obs.sz;

      m.color.r = 0.8;
      m.color.g = 0.1;
      m.color.b = 0.1;
      m.color.a = 0.75;

      array.markers.push_back(m);
    }
  }

  void addStartGoalMarkers(visualization_msgs::msg::MarkerArray & array)
  {
    visualization_msgs::msg::Marker start;
    start.header.frame_id = "map";
    start.header.stamp = this->now();
    start.ns = "start_goal";
    start.id = 1000;
    start.type = visualization_msgs::msg::Marker::SPHERE;
    start.action = visualization_msgs::msg::Marker::ADD;

    start.pose.position.x = start_pos_.x;
    start.pose.position.y = start_pos_.y;
    start.pose.position.z = start_pos_.z;

    start.scale.x = 0.35;
    start.scale.y = 0.35;
    start.scale.z = 0.35;

    start.color.r = 0.0;
    start.color.g = 1.0;
    start.color.b = 0.0;
    start.color.a = 1.0;

    visualization_msgs::msg::Marker goal = start;
    goal.id = 1001;
    goal.pose.position.x = goal_pos_.x;
    goal.pose.position.y = goal_pos_.y;
    goal.pose.position.z = goal_pos_.z;

    goal.color.r = 0.0;
    goal.color.g = 0.2;
    goal.color.b = 1.0;
    goal.color.a = 1.0;

    array.markers.push_back(start);
    array.markers.push_back(goal);
  }

  void addPathMarker(visualization_msgs::msg::MarkerArray & array)
  {
    visualization_msgs::msg::Marker path;
    path.header.frame_id = "map";
    path.header.stamp = this->now();
    path.ns = "kino_astar_path";
    path.id = 2000;
    path.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path.action = visualization_msgs::msg::Marker::ADD;

    path.scale.x = 0.08;

    path.color.r = 0.1;
    path.color.g = 0.9;
    path.color.b = 0.1;
    path.color.a = 1.0;

    path.points = kino_path_;

    array.markers.push_back(path);
  }

  double x_min_;
  double x_max_;
  double y_min_;
  double y_max_;
  double z_min_;
  double z_max_;

  double pos_resolution_;
  double vel_resolution_;

  int nx_;
  int ny_;
  int nz_;
  int nv_;

  double max_vel_;
  double max_acc_;
  double dt_;
  int primitive_check_num_;
  double goal_tolerance_;
  double safety_margin_;
  double max_search_time_;
  int max_expand_num_;

  Vec3 start_pos_;
  Vec3 start_vel_;
  Vec3 goal_pos_;

  std::vector<BoxObstacle> obstacles_;
  std::vector<geometry_msgs::msg::Point> kino_path_;
  std::vector<Vec3> kino_velocities_;
  std::vector<Vec3> kino_accelerations_;

  std::string scenario_;

  double planning_time_ms_ = 0.0;
  int expanded_nodes_ = 0;
  double traj_length_ = 0.0;
  double traj_duration_ = 0.0;
  double min_clearance_ = 0.0;
  double max_velocity_ = 0.0;
  double avg_velocity_ = 0.0;
  double max_acceleration_ = 0.0;
  bool collision_free_ = false;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KinoAStarDemoNode>());
  rclcpp::shutdown();
  return 0;
}
