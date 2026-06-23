// bspline_planner.cpp
//
// B-spline trajectory optimisation planner node for the Ackermann car.
// Pipeline: Kino A* front-end → cubic B-spline → ESDF-based optimisation.
// Publishes /map, /planner/path (optimised), /planner/kino_path (front-end).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"

#include "map_loader.hpp"

using namespace std::chrono_literals;

struct Vec3 { double x, y, z; };
struct AckermannState { double x, y, theta, v; };
struct KinoNode { AckermannState s; double time, g_score, f_score; int parent; long long key; };
struct QueueItem { double f_score; int node_id; bool operator<(const QueueItem & o) const { return f_score > o.f_score; } };

class BsplinePlannerNode : public rclcpp::Node
{
public:
  BsplinePlannerNode() : Node("bspline_planner")
  {
    map_yaml_ = this->declare_parameter("map_yaml", std::string(""));
    start_x_ = this->declare_parameter("start_x", -7.0);
    start_y_ = this->declare_parameter("start_y", -6.0);
    start_theta_ = this->declare_parameter("start_theta", std::atan2(6.0, 7.0));
    goal_x_ = this->declare_parameter("goal_x", 7.0);
    goal_y_ = this->declare_parameter("goal_y", 6.0);

    rclcpp::QoS map_qos(rclcpp::KeepLast(1));
    map_qos.transient_local();
    map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", map_qos);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planner/path", 1);
    kino_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planner/kino_path", 1);
    start_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/planner/start", 1);
    goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/planner/goal", 1);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Vehicle params
    wheelbase_ = 1.5; max_vel_ = 2.0; max_steer_ = 35.0 * M_PI / 180.0;
    kappa_max_ = std::tan(max_steer_) / wheelbase_;
    v_cruise_ = 1.5; dt_ = 0.4; primitive_check_ = 8;
    goal_tol_ = 0.65; goal_theta_tol_ = M_PI / 6.0; safety_margin_ = 0.3;
    max_time_ = 25.0; max_expand_ = 200000;
    pos_res_ = 0.4; theta_res_ = M_PI / 12.0; vel_res_ = 0.5;

    // B-spline params
    bspline_seg_ = 12; opt_iters_ = 400; opt_step_ = 0.06;
    lambda_s_ = 1.0; lambda_c_ = 2.5; lambda_k_ = 3.0; d_safe_ = 0.55;
    esdf_res_ = 0.12;

    if (map_yaml_.empty()) { RCLCPP_ERROR(this->get_logger(), "map_yaml not set!"); return; }
    if (!map_loader_.load(map_yaml_)) { RCLCPP_ERROR(this->get_logger(), "Failed to load map!"); return; }

    const auto & info = map_loader_.getInfo();
    x_min_ = info.origin_x; y_min_ = info.origin_y;
    x_max_ = info.origin_x + info.width * info.resolution;
    y_max_ = info.origin_y + info.height * info.resolution;
    nx_ = static_cast<int>((x_max_ - x_min_) / pos_res_);
    ny_ = static_cast<int>((y_max_ - y_min_) / pos_res_);
    ntheta_ = static_cast<int>(2.0 * M_PI / theta_res_);
    nv_ = static_cast<int>(2.0 * max_vel_ / vel_res_) + 1;

    RCLCPP_INFO(this->get_logger(), "Map loaded: %dx%d, res=%.3f", info.width, info.height, info.resolution);

    // Cache grid data for ESDF computation
    grid_data_ = map_loader_.getOccupancyGrid().data;

    if (runKinoAStar()) {
      generateControlPoints();
      buildEsdf();
      runBsplineOptimization();
    } else {
      RCLCPP_ERROR(this->get_logger(), "Front-end failed!");
    }

    timer_ = this->create_wall_timer(200ms, [this]() {
      publishMap(); publishPath(); publishStartGoal(); publishTF();
    });
    RCLCPP_INFO(this->get_logger(), "bspline_planner started.");
  }

private:
  // ---- Kino A* front-end ----
  double angleDiff(double a, double b) const { double d = a - b; while (d > M_PI) d -= 2*M_PI; while (d < -M_PI) d += 2*M_PI; return d; }

  bool isStateValid(const AckermannState & s) const {
    if (s.x < x_min_ || s.x >= x_max_ || s.y < y_min_ || s.y >= y_max_) return false;
    if (map_loader_.isOccupied(s.x, s.y, safety_margin_)) return false;
    return std::abs(s.v) <= max_vel_ + 1e-6;
  }

  long long stateKey(const AckermannState & s) const {
    int ix = static_cast<int>((s.x - x_min_) / pos_res_);
    int iy = static_cast<int>((s.y - y_min_) / pos_res_);
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_) return -1;
    double th = std::fmod(s.theta, 2*M_PI); if (th < 0) th += 2*M_PI;
    int ith = static_cast<int>(th / theta_res_);
    if (ith < 0 || ith >= ntheta_) return -1;
    int iv = static_cast<int>(std::round((s.v + max_vel_) / vel_res_));
    if (iv < 0 || iv >= nv_) return -1;
    long long k = ix; k = k*ny_ + iy; k = k*ntheta_ + ith; k = k*nv_ + iv; return k;
  }

  AckermannState propagate(const AckermannState & s, double vc, double de, double dt) const {
    double v = std::clamp(vc, -max_vel_, max_vel_), d = std::clamp(de, -max_steer_, max_steer_);
    double dth = v * std::tan(d) / wheelbase_ * dt;
    AckermannState ns;
    ns.x = s.x + v * std::cos(s.theta + 0.5*dth) * dt;
    ns.y = s.y + v * std::sin(s.theta + 0.5*dth) * dt;
    ns.theta = std::fmod(s.theta + dth, 2*M_PI); if (ns.theta < 0) ns.theta += 2*M_PI;
    ns.v = v; return ns;
  }

  bool checkPrimitive(const AckermannState & s0, double vc, double de) const {
    for (int i = 1; i <= primitive_check_; ++i)
      if (!isStateValid(propagate(s0, vc, de, dt_ * i / primitive_check_))) return false;
    return true;
  }

  double heuristic(const AckermannState & s) const {
    return std::hypot(s.x - goal_x_, s.y - goal_y_) / max_vel_ + 0.15 * std::abs(angleDiff(s.theta, std::atan2(goal_y_ - s.y, goal_x_ - s.x)));
  }

  bool runKinoAStar() {
    AckermannState start; start.x = start_x_; start.y = start_y_; start.theta = start_theta_; start.v = 0.0;
    if (!isStateValid(start)) return false;
    std::vector<KinoNode> nodes; std::priority_queue<QueueItem> open;
    std::unordered_map<long long, double> bg; std::unordered_map<long long, int> bi; std::unordered_set<long long> cl;
    long long sk = stateKey(start);
    KinoNode sn; sn.s = start; sn.time = 0; sn.g_score = 0; sn.f_score = heuristic(start); sn.parent = -1; sn.key = sk;
    nodes.push_back(sn); bg[sk] = 0; bi[sk] = 0; open.push({sn.f_score, 0});
    std::vector<std::pair<double,double>> cs;
    for (double v : {0.5*v_cruise_, v_cruise_}) for (double s : {-max_steer_, -0.5*max_steer_, 0.0, 0.5*max_steer_, max_steer_}) cs.push_back({v, s});
    cs.push_back({-0.4*v_cruise_, 0.0});
    int fid = -1, exp = 0;
    while (!open.empty() && exp < max_expand_) {
      auto it = open.top(); open.pop(); int ci = it.node_id; const KinoNode cur = nodes[ci];
      if (bi.find(cur.key) == bi.end() || bi[cur.key] != ci) continue; if (cl.count(cur.key)) continue; cl.insert(cur.key); exp++;
      if (std::hypot(cur.s.x - goal_x_, cur.s.y - goal_y_) < goal_tol_ && cur.time > 0.5 &&
          std::abs(angleDiff(cur.s.theta, std::atan2(goal_y_ - cur.s.y, goal_x_ - cur.s.x))) < goal_theta_tol_) { fid = ci; break; }
      if (cur.time > max_time_) continue;
      for (const auto & c : cs) {
        if (!checkPrimitive(cur.s, c.first, c.second)) continue;
        AckermannState ns = propagate(cur.s, c.first, c.second, dt_);
        long long nk = stateKey(ns); if (nk < 0 || cl.count(nk)) continue;
        double ng = cur.g_score + dt_ + 0.02 * std::abs(c.second) * dt_;
        auto gi = bg.find(nk); if (gi == bg.end() || ng < gi->second) {
          KinoNode nn; nn.s = ns; nn.time = cur.time + dt_; nn.g_score = ng; nn.f_score = ng + heuristic(ns); nn.parent = ci; nn.key = nk;
          int id = static_cast<int>(nodes.size()); nodes.push_back(nn); bg[nk] = ng; bi[nk] = id; open.push({nn.f_score, id});
        }
      }
    }
    kino_path_.clear(); if (fid < 0) return false;
    std::vector<int> ids; int id = fid; while (id >= 0) { ids.push_back(id); id = nodes[id].parent; }
    std::reverse(ids.begin(), ids.end());
    for (int nid : ids) { geometry_msgs::msg::PoseStamped p; p.header.frame_id = "map";
      p.pose.position.x = nodes[nid].s.x; p.pose.position.y = nodes[nid].s.y;
      tf2::Quaternion q; q.setRPY(0, 0, nodes[nid].s.theta);
      p.pose.orientation.x = q.x(); p.pose.orientation.y = q.y(); p.pose.orientation.z = q.z(); p.pose.orientation.w = q.w();
      kino_path_.push_back(p);
    }
    RCLCPP_INFO(this->get_logger(), "Kino A* front-end: %zu points", kino_path_.size());
    return true;
  }

  // ---- ESDF ----
  void buildEsdf() {
    esdf_nx_ = static_cast<int>((x_max_ - x_min_) / esdf_res_) + 1;
    esdf_ny_ = static_cast<int>((y_max_ - y_min_) / esdf_res_) + 1;
    esdf_.assign(esdf_nx_ * esdf_ny_, 0.0);
    for (int ix = 0; ix < esdf_nx_; ++ix) {
      double x = x_min_ + ix * esdf_res_;
      for (int iy = 0; iy < esdf_ny_; ++iy) {
        double y = y_min_ + iy * esdf_res_;
        double best = 1e9;
        // Check grid cells for obstacles
        int gx, gy;
        if (map_loader_.worldToGrid(x, y, gx, gy)) {
          const auto & grid = map_loader_.getOccupancyGrid();
          int idx = gy * map_loader_.getInfo().width + gx;
          if (grid.data[idx] > 50) { esdf_[iy * esdf_nx_ + ix] = 0.0; continue; }
        }
        // Compute distance to nearest occupied cell (brute force, limited range)
        int mr = static_cast<int>(d_safe_ / map_loader_.getInfo().resolution) + 1;
        const auto & info = map_loader_.getInfo();
        for (int dx = -mr; dx <= mr; ++dx) {
          for (int dy = -mr; dy <= mr; ++dy) {
            int cx = gx + dx, cy = gy + dy;
            if (cx < 0 || cx >= info.width || cy < 0 || cy >= info.height) continue;
            if (grid_data_[cy * info.width + cx] > 50) {
              double wx, wy; map_loader_.gridToWorld(cx, cy, wx, wy);
              double d = std::hypot(wx - x, wy - y);
              best = std::min(best, d);
            }
          }
        }
        esdf_[iy * esdf_nx_ + ix] = best;
      }
    }
  }

  double esdfVal(double x, double y) const {
    double fx = (x - x_min_) / esdf_res_, fy = (y - y_min_) / esdf_res_;
    int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, esdf_nx_ - 2);
    int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, esdf_ny_ - 2);
    double tx = std::clamp(fx - ix, 0.0, 1.0), ty = std::clamp(fy - iy, 0.0, 1.0);
    return (esdf_[iy*esdf_nx_+ix]*(1-tx) + esdf_[iy*esdf_nx_+ix+1]*tx)*(1-ty) +
           (esdf_[(iy+1)*esdf_nx_+ix]*(1-tx) + esdf_[(iy+1)*esdf_nx_+ix+1]*tx)*ty;
  }

  void esdfGrad(double x, double y, double & gx, double & gy) const {
    double h = esdf_res_;
    gx = (esdfVal(x+h, y) - esdfVal(x-h, y)) / (2*h);
    gy = (esdfVal(x, y+h) - esdfVal(x, y-h)) / (2*h);
  }

  // ---- B-spline ----
  void generateControlPoints() {
    cps_.clear(); if (kino_path_.size() < 2) return;
    auto toPt = [](const geometry_msgs::msg::PoseStamped & p) { return p.pose.position; };
    cps_.push_back(toPt(kino_path_.front())); cps_.push_back(toPt(kino_path_.front()));
    for (const auto & p : kino_path_) cps_.push_back(toPt(p));
    cps_.push_back(toPt(kino_path_.back())); cps_.push_back(toPt(kino_path_.back()));
  }

  geometry_msgs::msg::Point splinePt(const geometry_msgs::msg::Point & p0, const geometry_msgs::msg::Point & p1,
                                     const geometry_msgs::msg::Point & p2, const geometry_msgs::msg::Point & p3, double u) const {
    double u2 = u*u, u3 = u2*u;
    geometry_msgs::msg::Point p;
    p.x = ((1-3*u+3*u2-u3)*p0.x + (4-6*u2+3*u3)*p1.x + (1+3*u+3*u2-3*u3)*p2.x + u3*p3.x) / 6.0;
    p.y = ((1-3*u+3*u2-u3)*p0.y + (4-6*u2+3*u3)*p1.y + (1+3*u+3*u2-3*u3)*p2.y + u3*p3.y) / 6.0;
    p.z = 0.0; return p;
  }

  std::vector<geometry_msgs::msg::Point> sample(const std::vector<geometry_msgs::msg::Point> & q) const {
    std::vector<geometry_msgs::msg::Point> path;
    if (q.size() < 4) return path;
    for (size_t i = 0; i + 3 < q.size(); ++i)
      for (int j = 0; j < bspline_seg_; ++j)
        path.push_back(splinePt(q[i], q[i+1], q[i+2], q[i+3], static_cast<double>(j)/bspline_seg_));
    path.push_back(q.back()); return path;
  }

  double maxCurvature(const std::vector<geometry_msgs::msg::Point> & path) const {
    if (path.size() < 3) return 0.0; double mc = 0.0;
    for (size_t i = 1; i + 1 < path.size(); ++i) {
      double dx = path[i+1].x - path[i-1].x, dy = path[i+1].y - path[i-1].y;
      double ddx = path[i+1].x - 2*path[i].x + path[i-1].x, ddy = path[i+1].y - 2*path[i].y + path[i-1].y;
      mc = std::max(mc, std::abs(dx*ddy - dy*ddx) / (std::pow(dx*dx+dy*dy, 1.5) + 1e-6));
    }
    return mc;
  }

  void runBsplineOptimization() {
    auto t0 = std::chrono::steady_clock::now();
    auto q = cps_;
    for (int iter = 0; iter < opt_iters_; ++iter) {
      std::vector<Vec3> grad(q.size(), Vec3{0,0,0});
      // Smoothness
      for (size_t i = 1; i + 1 < q.size(); ++i) {
        Vec3 a{q[i+1].x - 2*q[i].x + q[i-1].x, q[i+1].y - 2*q[i].y + q[i-1].y, 0};
        grad[i-1].x += lambda_s_ * 2 * a.x; grad[i-1].y += lambda_s_ * 2 * a.y;
        grad[i].x   -= lambda_s_ * 4 * a.x; grad[i].y   -= lambda_s_ * 4 * a.y;
        grad[i+1].x += lambda_s_ * 2 * a.x; grad[i+1].y += lambda_s_ * 2 * a.y;
      }
      // Collision (ESDF)
      for (size_t i = 0; i < q.size(); ++i) {
        double d = esdfVal(q[i].x, q[i].y);
        if (d >= d_safe_) continue;
        double gx, gy; esdfGrad(q[i].x, q[i].y, gx, gy);
        double coef = -2.0 * lambda_c_ * (d_safe_ - d);
        grad[i].x += coef * gx; grad[i].y += coef * gy;
      }
      // Curvature penalty
      for (size_t i = 1; i + 1 < q.size(); ++i) {
        double dx = q[i+1].x - q[i-1].x, dy = q[i+1].y - q[i-1].y;
        double ddx = q[i+1].x - 2*q[i].x + q[i-1].x, ddy = q[i+1].y - 2*q[i].y + q[i-1].y;
        double num = dx*ddy - dy*ddx, den = std::pow(dx*dx+dy*dy, 1.5) + 1e-6;
        double k = num / den, over = std::abs(k) - kappa_max_;
        if (over <= 0.0) continue;
        double sign = (k > 0 ? 1.0 : -1.0), gk = 2.0 * lambda_k_ * over * sign;
        for (int kk = 0; kk < 3; ++kk) {
          double e = 1e-4; auto qq = q;
          if (kk == 0) qq[i-1].x += e; else if (kk == 1) qq[i].x += e; else qq[i+1].x += e;
          double dx2 = qq[i+1].x - qq[i-1].x, ddx2 = qq[i+1].x - 2*qq[i].x + qq[i-1].x;
          double k2x = (dx2*ddy - dy*ddx2) / std::pow(dx2*dx2+dy*dy, 1.5);
          if (kk == 0) grad[i-1].x += gk * (k2x - k) / e;
          else if (kk == 1) grad[i].x += gk * (k2x - k) / e;
          else grad[i+1].x += gk * (k2x - k) / e;
        }
      }
      for (size_t i = 2; i + 2 < q.size(); ++i) {
        q[i].x -= opt_step_ * grad[i].x; q[i].y -= opt_step_ * grad[i].y;
        q[i].x = std::clamp(q[i].x, x_min_ + 0.2, x_max_ - 0.2);
        q[i].y = std::clamp(q[i].y, y_min_ + 0.2, y_max_ - 0.2); q[i].z = 0.0;
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    opt_path_ = sample(q);
    RCLCPP_INFO(this->get_logger(), "B-spline opt: %.1f ms, iters: %d, path pts: %zu", ms, opt_iters_, opt_path_.size());
    RCLCPP_INFO(this->get_logger(), "Max curvature: %.4f (limit %.4f)", maxCurvature(opt_path_), kappa_max_);
  }

  // ---- Publishing ----
  void publishMap() {
    auto grid = map_loader_.getOccupancyGrid();
    grid.header.stamp = this->now(); grid.header.frame_id = "map";
    map_pub_->publish(grid);
  }

  void publishPath() {
    if (!opt_path_.empty()) {
      nav_msgs::msg::Path msg; msg.header.stamp = this->now(); msg.header.frame_id = "map";
      for (const auto & p : opt_path_) { geometry_msgs::msg::PoseStamped ps; ps.header = msg.header; ps.pose.position = p; ps.pose.orientation.w = 1.0; msg.poses.push_back(ps); }
      path_pub_->publish(msg);
    }
    if (!kino_path_.empty()) {
      nav_msgs::msg::Path kmsg;
      kmsg.header.stamp = this->now();
      kmsg.header.frame_id = "map";
      kmsg.poses = kino_path_;
      kino_path_pub_->publish(kmsg);
    }
  }

  void publishStartGoal() {
    geometry_msgs::msg::PoseStamped start, goal;
    start.header.stamp = this->now(); start.header.frame_id = "map";
    start.pose.position.x = start_x_; start.pose.position.y = start_y_;
    tf2::Quaternion q; q.setRPY(0, 0, start_theta_);
    start.pose.orientation.x = q.x(); start.pose.orientation.y = q.y(); start.pose.orientation.z = q.z(); start.pose.orientation.w = q.w();
    start_pub_->publish(start);
    goal.header = start.header; goal.pose.position.x = goal_x_; goal.pose.position.y = goal_y_; goal.pose.orientation.w = 1.0;
    goal_pub_->publish(goal);
  }

  void publishTF() {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now(); t.header.frame_id = "map"; t.child_frame_id = "odom";
    t.transform.rotation.w = 1.0; tf_broadcaster_->sendTransform(t);
    t.header.frame_id = "odom"; t.child_frame_id = "base_link";
    t.transform.translation.x = start_x_; t.transform.translation.y = start_y_;
    tf2::Quaternion q; q.setRPY(0, 0, start_theta_);
    t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y(); t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(t);
  }

  std::string map_yaml_;
  double start_x_, start_y_, start_theta_, goal_x_, goal_y_;
  double wheelbase_, max_vel_, max_steer_, kappa_max_, v_cruise_, dt_;
  int primitive_check_; double goal_tol_, goal_theta_tol_, safety_margin_, max_time_;
  int max_expand_; double pos_res_, theta_res_, vel_res_;
  int nx_, ny_, ntheta_, nv_; double x_min_, x_max_, y_min_, y_max_;
  int bspline_seg_, opt_iters_; double opt_step_, lambda_s_, lambda_c_, lambda_k_, d_safe_, esdf_res_;
  int esdf_nx_, esdf_ny_; std::vector<double> esdf_;

  MapLoader map_loader_;
  std::vector<int8_t> grid_data_;
  std::vector<geometry_msgs::msg::PoseStamped> kino_path_;
  std::vector<geometry_msgs::msg::Point> cps_, opt_path_;

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_, kino_path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr start_pub_, goal_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BsplinePlannerNode>());
  rclcpp::shutdown();
  return 0;
}