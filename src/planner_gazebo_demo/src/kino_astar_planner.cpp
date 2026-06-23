// kino_astar_planner.cpp
//
// Kinodynamic A* planner node for the Ackermann car.
// Loads a PGM+YAML grid map, runs the search, and publishes:
//   /map             — nav_msgs/OccupancyGrid
//   /planner/path    — nav_msgs/Path
//   /planner/start   — geometry_msgs/PoseStamped
//   /planner/goal    — geometry_msgs/PoseStamped
//
// Visualised in RViz with the Map, Path, and TF displays (no Markers).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"

#include "map_loader.hpp"

using namespace std::chrono_literals;

struct AckermannState
{
  double x, y, theta, v;
};

struct KinoNode
{
  AckermannState s;
  double time, g_score, f_score;
  int parent;
  long long key;
};

struct QueueItem
{
  double f_score;
  int node_id;
  bool operator<(const QueueItem & o) const { return f_score > o.f_score; }
};

class KinoAstarPlannerNode : public rclcpp::Node
{
public:
  KinoAstarPlannerNode() : Node("kino_astar_planner")
  {
    // Declare parameters
    map_yaml_ = this->declare_parameter("map_yaml", std::string(""));
    start_x_ = this->declare_parameter("start_x", -7.0);
    start_y_ = this->declare_parameter("start_y", -6.0);
    start_theta_ = this->declare_parameter("start_theta", std::atan2(6.0, 7.0));
    goal_x_ = this->declare_parameter("goal_x", 7.0);
    goal_y_ = this->declare_parameter("goal_y", 6.0);

    // Publishers — map uses transient_local (latched) QoS so RViz receives it
    rclcpp::QoS map_qos(rclcpp::KeepLast(1));
    map_qos.transient_local();
    map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", map_qos);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planner/path", 1);
    start_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/planner/start", 1);
    goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/planner/goal", 1);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Vehicle params
    wheelbase_ = 1.5;
    max_vel_ = 2.0;
    max_steer_ = 35.0 * M_PI / 180.0;
    v_cruise_ = 1.5;
    dt_ = 0.4;
    primitive_check_ = 8;
    goal_tol_ = 0.65;
    goal_theta_tol_ = M_PI / 6.0;
    safety_margin_ = 0.3;
    max_time_ = 25.0;
    max_expand_ = 200000;
    pos_res_ = 0.4;
    theta_res_ = M_PI / 12.0;
    vel_res_ = 0.5;
    nx_ = 0; ny_ = 0; ntheta_ = 0; nv_ = 0;

    // Load map
    if (map_yaml_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "map_yaml parameter not set!");
      return;
    }
    if (!map_loader_.load(map_yaml_)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load map from %s", map_yaml_.c_str());
      return;
    }

    const auto & info = map_loader_.getInfo();
    x_min_ = info.origin_x;
    y_min_ = info.origin_y;
    x_max_ = info.origin_x + info.width * info.resolution;
    y_max_ = info.origin_y + info.height * info.resolution;
    nx_ = static_cast<int>((x_max_ - x_min_) / pos_res_);
    ny_ = static_cast<int>((y_max_ - y_min_) / pos_res_);
    ntheta_ = static_cast<int>(2.0 * M_PI / theta_res_);
    nv_ = static_cast<int>(2.0 * max_vel_ / vel_res_) + 1;

    RCLCPP_INFO(this->get_logger(), "Map loaded: %dx%d, res=%.3f, origin=(%.1f,%.1f)",
      info.width, info.height, info.resolution, info.origin_x, info.origin_y);
    RCLCPP_INFO(this->get_logger(), "Bounds: x=[%.1f,%.1f] y=[%.1f,%.1f]",
      x_min_, x_max_, y_min_, y_max_);

    // Run planner
    runPlanner();

    // Latched map + periodic path/TF
    timer_ = this->create_wall_timer(200ms, [this]() {
      publishMap();
      publishPath();
      publishStartGoal();
      publishTF();
    });

    RCLCPP_INFO(this->get_logger(), "kino_astar_planner started.");
  }

private:
  double angleDiff(double a, double b) const
  {
    double d = a - b;
    while (d > M_PI) d -= 2 * M_PI;
    while (d < -M_PI) d += 2 * M_PI;
    return d;
  }

  bool isStateValid(const AckermannState & s) const
  {
    if (s.x < x_min_ || s.x >= x_max_ || s.y < y_min_ || s.y >= y_max_)
      return false;
    if (map_loader_.isOccupied(s.x, s.y, safety_margin_))
      return false;
    if (s.v < -max_vel_ - 1e-6 || s.v > max_vel_ + 1e-6)
      return false;
    return true;
  }

  long long stateKey(const AckermannState & s) const
  {
    int ix = static_cast<int>((s.x - x_min_) / pos_res_);
    int iy = static_cast<int>((s.y - y_min_) / pos_res_);
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_) return -1;
    double th = std::fmod(s.theta, 2 * M_PI);
    if (th < 0) th += 2 * M_PI;
    int ith = static_cast<int>(th / theta_res_);
    if (ith < 0 || ith >= ntheta_) return -1;
    int iv = static_cast<int>(std::round((s.v + max_vel_) / vel_res_));
    if (iv < 0 || iv >= nv_) return -1;
    long long k = ix;
    k = k * ny_ + iy;
    k = k * ntheta_ + ith;
    k = k * nv_ + iv;
    return k;
  }

  AckermannState propagate(const AckermannState & s, double vc, double de, double dt) const
  {
    double v = std::clamp(vc, -max_vel_, max_vel_);
    double d = std::clamp(de, -max_steer_, max_steer_);
    double dth = v * std::tan(d) / wheelbase_ * dt;
    AckermannState ns;
    ns.x = s.x + v * std::cos(s.theta + 0.5 * dth) * dt;
    ns.y = s.y + v * std::sin(s.theta + 0.5 * dth) * dt;
    ns.theta = std::fmod(s.theta + dth, 2 * M_PI);
    if (ns.theta < 0) ns.theta += 2 * M_PI;
    ns.v = v;
    return ns;
  }

  bool checkPrimitive(const AckermannState & s0, double vc, double de) const
  {
    for (int i = 1; i <= primitive_check_; ++i) {
      if (!isStateValid(propagate(s0, vc, de, dt_ * i / primitive_check_)))
        return false;
    }
    return true;
  }

  double heuristic(const AckermannState & s) const
  {
    double d = std::hypot(s.x - goal_x_, s.y - goal_y_);
    double dh = std::atan2(goal_y_ - s.y, goal_x_ - s.x);
    return d / max_vel_ + 0.15 * std::abs(angleDiff(s.theta, dh));
  }

  void runPlanner()
  {
    AckermannState start;
    start.x = start_x_; start.y = start_y_;
    start.theta = start_theta_; start.v = 0.0;

    if (!isStateValid(start)) {
      RCLCPP_ERROR(this->get_logger(), "Start state invalid!");
      return;
    }

    std::vector<std::pair<double, double>> controls;
    for (double v : {0.5 * v_cruise_, v_cruise_})
      for (double s : {-max_steer_, -0.5 * max_steer_, 0.0, 0.5 * max_steer_, max_steer_})
        controls.push_back({v, s});
    controls.push_back({-0.4 * v_cruise_, 0.0});

    std::vector<KinoNode> nodes;
    std::priority_queue<QueueItem> open;
    std::unordered_map<long long, double> best_g;
    std::unordered_map<long long, int> best_id;
    std::unordered_set<long long> closed;

    long long sk = stateKey(start);
    KinoNode sn;
    sn.s = start; sn.time = 0; sn.g_score = 0;
    sn.f_score = heuristic(start); sn.parent = -1; sn.key = sk;
    nodes.push_back(sn);
    best_g[sk] = 0; best_id[sk] = 0;
    open.push({sn.f_score, 0});

    int fid = -1, exp = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (!open.empty() && exp < max_expand_) {
      auto it = open.top(); open.pop();
      int ci = it.node_id;
      const KinoNode cur = nodes[ci];
      if (best_id.find(cur.key) == best_id.end() || best_id[cur.key] != ci) continue;
      if (closed.count(cur.key)) continue;
      closed.insert(cur.key); exp++;

      if (std::hypot(cur.s.x - goal_x_, cur.s.y - goal_y_) < goal_tol_ && cur.time > 0.5 &&
          std::abs(angleDiff(cur.s.theta, std::atan2(goal_y_ - cur.s.y, goal_x_ - cur.s.x))) < goal_theta_tol_) {
        fid = ci; break;
      }
      if (cur.time > max_time_) continue;

      for (const auto & c : controls) {
        if (!checkPrimitive(cur.s, c.first, c.second)) continue;
        AckermannState ns = propagate(cur.s, c.first, c.second, dt_);
        long long nk = stateKey(ns);
        if (nk < 0 || closed.count(nk)) continue;
        double ng = cur.g_score + dt_ + 0.02 * std::abs(c.second) * dt_;
        auto gi = best_g.find(nk);
        if (gi == best_g.end() || ng < gi->second) {
          KinoNode nn;
          nn.s = ns; nn.time = cur.time + dt_;
          nn.g_score = ng; nn.f_score = ng + heuristic(ns);
          nn.parent = ci; nn.key = nk;
          int id = static_cast<int>(nodes.size());
          nodes.push_back(nn);
          best_g[nk] = ng; best_id[nk] = id;
          open.push({nn.f_score, id});
        }
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    path_.clear();
    if (fid < 0) {
      RCLCPP_ERROR(this->get_logger(), "Kino A* failed! Expanded: %d, time: %.1f ms", exp, ms);
      return;
    }

    std::vector<int> ids;
    int id = fid;
    while (id >= 0) { ids.push_back(id); id = nodes[id].parent; }
    std::reverse(ids.begin(), ids.end());
    for (int nid : ids) {
      geometry_msgs::msg::PoseStamped p;
      p.header.frame_id = "map";
      p.pose.position.x = nodes[nid].s.x;
      p.pose.position.y = nodes[nid].s.y;
      p.pose.position.z = 0.0;
      tf2::Quaternion q;
      q.setRPY(0, 0, nodes[nid].s.theta);
      p.pose.orientation.x = q.x(); p.pose.orientation.y = q.y();
      p.pose.orientation.z = q.z(); p.pose.orientation.w = q.w();
      path_.push_back(p);
    }

    RCLCPP_INFO(this->get_logger(), "Kino A* success! Expanded: %d, path points: %zu, time: %.1f ms",
      exp, path_.size(), ms);
  }

  void publishMap()
  {
    auto grid = map_loader_.getOccupancyGrid();
    grid.header.stamp = this->now();
    grid.header.frame_id = "map";
    map_pub_->publish(grid);
  }

  void publishPath()
  {
    if (path_.empty()) return;
    nav_msgs::msg::Path msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "map";
    msg.poses = path_;
    path_pub_->publish(msg);
  }

  void publishStartGoal()
  {
    geometry_msgs::msg::PoseStamped start, goal;
    start.header.stamp = this->now();
    start.header.frame_id = "map";
    start.pose.position.x = start_x_;
    start.pose.position.y = start_y_;
    tf2::Quaternion q;
    q.setRPY(0, 0, start_theta_);
    start.pose.orientation.x = q.x(); start.pose.orientation.y = q.y();
    start.pose.orientation.z = q.z(); start.pose.orientation.w = q.w();
    start_pub_->publish(start);

    goal.header = start.header;
    goal.pose.position.x = goal_x_;
    goal.pose.position.y = goal_y_;
    goal.pose.orientation.w = 1.0;
    goal_pub_->publish(goal);
  }

  void publishTF()
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = "map";
    t.child_frame_id = "odom";
    t.transform.rotation.w = 1.0;
    tf_broadcaster_->sendTransform(t);

    t.header.frame_id = "odom";
    t.child_frame_id = "base_link";
    t.transform.translation.x = start_x_;
    t.transform.translation.y = start_y_;
    tf2::Quaternion q;
    q.setRPY(0, 0, start_theta_);
    t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(t);
  }

  // Params
  std::string map_yaml_;
  double start_x_, start_y_, start_theta_, goal_x_, goal_y_;
  double wheelbase_, max_vel_, max_steer_, v_cruise_, dt_;
  int primitive_check_;
  double goal_tol_, goal_theta_tol_, safety_margin_, max_time_;
  int max_expand_;
  double pos_res_, theta_res_, vel_res_;
  int nx_, ny_, ntheta_, nv_;
  double x_min_, x_max_, y_min_, y_max_;

  MapLoader map_loader_;
  std::vector<geometry_msgs::msg::PoseStamped> path_;

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr start_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KinoAstarPlannerNode>());
  rclcpp::shutdown();
  return 0;
}