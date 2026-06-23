// path_player.cpp
//
// Animates the car driving along the planned path in RViz (stage 2 has no
// Gazebo/controllers, so nothing actually moves the robot). It:
//   - subscribes to /planner/path (nav_msgs/Path),
//   - walks a point along the path at a slow constant speed,
//   - broadcasts the dynamic map->base_link transform (so the car moves),
//   - publishes /joint_states with the front-wheel STEERING angle (from the path
//     curvature) and the rolling spin of all four wheels (so you can see the
//     wheels turn and steer),
//   - loops back to the start when it reaches the end.
//
// This REPLACES the static map->base_link publisher and joint_state_publisher in
// planner.launch.py when animate:=true. Before a path arrives it just parks the
// car at the start pose so the TF tree is always complete.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"

class PathPlayer : public rclcpp::Node
{
public:
  PathPlayer() : Node("path_player")
  {
    speed_         = this->declare_parameter("play_speed", 0.6);     // m/s — slow
    wheelbase_     = this->declare_parameter("wheelbase", 0.9);
    wheel_radius_  = this->declare_parameter("wheel_radius", 0.12);
    steer_max_     = this->declare_parameter("steer_max", 0.6109);
    loop_          = this->declare_parameter("loop", true);
    map_frame_     = this->declare_parameter("map_frame", std::string("map"));
    base_frame_    = this->declare_parameter("base_frame", std::string("base_link"));
    start_x_       = this->declare_parameter("start_x", -7.0);
    start_y_       = this->declare_parameter("start_y", -6.0);
    start_yaw_     = this->declare_parameter("start_yaw", 0.706);

    tfb_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    js_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    // Planner publishes /planner/path VOLATILE (and re-publishes it every 200 ms),
    // so subscribe volatile to stay QoS-compatible.
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      "/planner/path", rclcpp::QoS(10),
      [this](nav_msgs::msg::Path::SharedPtr m) {setPath(*m);});

    dt_ = 0.033;   // ~30 Hz
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(33), std::bind(&PathPlayer::tick, this));

    RCLCPP_INFO(this->get_logger(),
      "path_player: animating /planner/path at %.2f m/s (loop=%s).",
      speed_, loop_ ? "true" : "false");
  }

private:
  static double wrap(double a)
  {
    while (a > M_PI) {a -= 2.0 * M_PI;}
    while (a < -M_PI) {a += 2.0 * M_PI;}
    return a;
  }

  void setPath(const nav_msgs::msg::Path & p)
  {
    if (p.poses.size() < 2) {return;}

    // The planner re-publishes the SAME path every 200 ms. Only (re)load it when
    // it actually changed, otherwise we'd reset s_ each time and the car would
    // never progress (it would keep restarting at the path start).
    const bool same =
      have_path_ && p.poses.size() == xs_.size() &&
      std::abs(p.poses.front().pose.position.x - xs_.front()) < 1e-3 &&
      std::abs(p.poses.back().pose.position.x - xs_.back()) < 1e-3 &&
      std::abs(p.poses.back().pose.position.y - ys_.back()) < 1e-3;
    if (same) {return;}

    xs_.clear(); ys_.clear(); cum_.clear();
    for (const auto & ps : p.poses) {
      xs_.push_back(ps.pose.position.x);
      ys_.push_back(ps.pose.position.y);
    }
    cum_.assign(xs_.size(), 0.0);
    for (size_t i = 1; i < xs_.size(); ++i) {
      cum_[i] = cum_[i - 1] + std::hypot(xs_[i] - xs_[i - 1], ys_[i] - ys_[i - 1]);
    }
    total_ = cum_.back();
    s_ = 0.0;
    have_path_ = total_ > 1e-3;
  }

  // Pose (x, y, heading) at arc length s along the path.
  void poseAt(double s, double & x, double & y, double & th) const
  {
    s = std::clamp(s, 0.0, total_);
    size_t i = 0;
    while (i + 1 < cum_.size() && cum_[i + 1] < s) {++i;}
    const size_t j = std::min(i + 1, xs_.size() - 1);
    const double seg = std::max(cum_[j] - cum_[i], 1e-6);
    const double t = std::clamp((s - cum_[i]) / seg, 0.0, 1.0);
    x = xs_[i] + t * (xs_[j] - xs_[i]);
    y = ys_[i] + t * (ys_[j] - ys_[i]);
    th = std::atan2(ys_[j] - ys_[i], xs_[j] - xs_[i]);
  }

  void tick()
  {
    double x, y, th, steer = 0.0;
    if (have_path_) {
      poseAt(s_, x, y, th);
      // Steering from local path curvature: delta = atan(L * dtheta/ds).
      const double ds = 0.4;
      double x2, y2, th2;
      poseAt(std::min(s_ + ds, total_), x2, y2, th2);
      const double dtheta = wrap(th2 - th);
      steer = std::clamp(std::atan(wheelbase_ * dtheta / ds), -steer_max_, steer_max_);
      // advance, loop at the end
      s_ += speed_ * dt_;
      if (s_ >= total_) {s_ = loop_ ? 0.0 : total_;}
      spin_ += (speed_ * dt_) / wheel_radius_;
    } else {
      // No path yet: park at the start pose.
      x = start_x_; y = start_y_; th = start_yaw_;
    }

    const rclcpp::Time now = this->get_clock()->now();

    // map -> base_link
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = now;
    tf.header.frame_id = map_frame_;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = x;
    tf.transform.translation.y = y;
    tf.transform.translation.z = 0.0;
    tf.transform.rotation.z = std::sin(th / 2.0);
    tf.transform.rotation.w = std::cos(th / 2.0);
    tfb_->sendTransform(tf);

    // joint states: front wheels steer, all four spin
    sensor_msgs::msg::JointState js;
    js.header.stamp = now;
    js.name = {
      "front_left_steering_joint", "front_right_steering_joint",
      "front_left_wheel_joint", "front_right_wheel_joint",
      "rear_left_wheel_joint", "rear_right_wheel_joint"};
    js.position = {steer, steer, spin_, spin_, spin_, spin_};
    js_pub_->publish(js);
  }

  std::shared_ptr<tf2_ros::TransformBroadcaster> tfb_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<double> xs_, ys_, cum_;
  double total_ = 0.0, s_ = 0.0, spin_ = 0.0, dt_ = 0.033;
  bool have_path_ = false;

  double speed_, wheelbase_, wheel_radius_, steer_max_, start_x_, start_y_, start_yaw_;
  bool loop_;
  std::string map_frame_, base_frame_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPlayer>());
  rclcpp::shutdown();
  return 0;
}
