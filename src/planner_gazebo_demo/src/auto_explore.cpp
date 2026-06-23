// auto_explore.cpp
//
// Hands-free exploration for SLAM mapping, anchored at the start point.
//
// Behaviour (a "fan of spokes" from the start):
//   - the steering controller's odometry starts at (0,0) at the spawn point, so
//     d = hypot(odom_x, odom_y) is exactly the distance from the start — no
//     coordinate conversion needed;
//   - OUTBOUND: drive away from the start along the current spoke heading until
//     d >= max_range (or the car gets stuck against something);
//   - INBOUND: turn around and drive back toward the start until d <= min_range;
//   - then rotate to the NEXT spoke heading and go out again.
//   Repeating this sweeps a fan of directions out of the start and back, so the
//   car never runs off in one direction and never leaves a bounded disk of
//   radius max_range around the start. The spokes point into the arena (the car
//   spawns facing the arena, i.e. odom +x), within +/- fan_half.
//
// SLOW constant speed (clean scans). Ackermann pure-pursuit steering so every
// command respects the car's minimum turning radius (no orbiting / "打转").
// Commands: geometry_msgs/Twist on /ackermann_steering_controller/reference_unstamped.
//
//   ros2 run planner_gazebo_demo auto_explore
//   ros2 run planner_gazebo_demo auto_explore --ros-args -p max_range:=8.0 -p cruise_speed:=0.4

#include <algorithm>
#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"

class AutoExplore : public rclcpp::Node
{
public:
  AutoExplore() : Node("auto_explore")
  {
    cruise_        = this->declare_parameter("cruise_speed", 0.4);   // m/s — keep SLOW (clean scans)
    max_range_     = this->declare_parameter("max_range", 9.0);      // m: turn back when this far from start
    min_range_     = this->declare_parameter("min_range", 2.0);      // m: "back at start" radius
    // Spokes span `sweep_deg` around the start. Default 360 -> full circle, i.e.
    // the car heads out in every direction, not just a forward fan. Set < 360 for
    // a fan centred on center_deg (0 = straight into the arena = odom +x).
    sweep_deg_     = this->declare_parameter("sweep_deg", 360.0);
    center_deg_    = this->declare_parameter("center_deg", 0.0);
    num_spokes_    = this->declare_parameter("num_spokes", 8);
    wheelbase_     = this->declare_parameter("wheelbase", 0.9);
    steer_max_     = this->declare_parameter("steer_max", 0.6109);   // 35 deg
    min_lookahead_ = this->declare_parameter("min_lookahead", 1.5);
    // Spawn pose (only used to print the car's world position in the debug log).
    spawn_x_       = this->declare_parameter("spawn_x", -7.0);
    spawn_y_       = this->declare_parameter("spawn_y", -6.0);
    spawn_yaw_     = this->declare_parameter("spawn_yaw", 0.706);

    pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/ackermann_steering_controller/reference_unstamped", 10);
    sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/ackermann_steering_controller/odometry", 10,
      [this](nav_msgs::msg::Odometry::SharedPtr m) {odom_ = m; have_odom_ = true;});
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&AutoExplore::tick, this));

    theta_out_ = spokeAngle(0);
    RCLCPP_INFO(this->get_logger(),
      "auto_explore: spoke explorer. max_range=%.1f m, %d spokes over %.0f deg, cruise=%.2f m/s.",
      max_range_, num_spokes_, sweep_deg_, cruise_);
  }

private:
  // Spoke heading (odom frame; 0 = straight into the arena = odom +x), index i.
  double spokeAngle(int i) const
  {
    const double center = center_deg_ * M_PI / 180.0;
    if (num_spokes_ <= 1) {return center;}
    if (sweep_deg_ >= 359.9) {
      // Full circle: evenly spaced over 360 deg (no overlapping endpoints).
      return wrap(center + i * (2.0 * M_PI / num_spokes_));
    }
    // Fan: span sweep_deg, centred on center.
    const double h = (sweep_deg_ * M_PI / 180.0) / 2.0;
    const double frac = double(i) / double(num_spokes_ - 1);   // 0..1
    return wrap(center - h + frac * (2.0 * h));
  }
  static double wrap(double a)
  {
    while (a > M_PI) {a -= 2.0 * M_PI;}
    while (a < -M_PI) {a += 2.0 * M_PI;}
    return a;
  }
  static double yawFromQuat(const geometry_msgs::msg::Quaternion & q)
  {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  void tick()
  {
    geometry_msgs::msg::Twist cmd;
    if (!have_odom_) {pub_->publish(cmd); return;}

    const double ox = odom_->pose.pose.position.x;
    const double oy = odom_->pose.pose.position.y;
    const double oyaw = yawFromQuat(odom_->pose.pose.orientation);
    const double d = std::hypot(ox, oy);   // distance from start (odom origin)

    // Anti-stuck: every ~3 s, if we barely moved, force a phase change.
    bool stuck = false;
    if (++stuck_ticks_ >= 60) {
      stuck = (std::hypot(ox - last_x_, oy - last_y_) < 0.25);
      last_x_ = ox; last_y_ = oy; stuck_ticks_ = 0;
    }

    // --- Phase transitions (the part you asked for) ---
    if (outbound_) {
      if (d >= max_range_ || stuck) {
        outbound_ = false;                       // too far (or blocked) -> head back
      }
    } else {
      if (d <= min_range_ || stuck) {
        spoke_ = (spoke_ + 1) % num_spokes_;      // back home -> next direction
        theta_out_ = spokeAngle(spoke_);
        outbound_ = true;
      }
    }

    // --- Choose the target point in the odom frame ---
    double tx, ty;
    if (outbound_) {
      tx = (max_range_ + 2.0) * std::cos(theta_out_);   // a point out along the spoke
      ty = (max_range_ + 2.0) * std::sin(theta_out_);
    } else {
      tx = 0.0; ty = 0.0;                                // the start
    }

    // --- Ackermann pure-pursuit steering toward the target ---
    const double alpha = wrap(std::atan2(ty - oy, tx - ox) - oyaw);
    double steer;
    if (std::abs(alpha) > 1.6) {
      steer = (alpha > 0.0) ? steer_max_ : -steer_max_;  // target behind -> full lock to swing around
    } else {
      const double Ld = std::max(std::hypot(tx - ox, ty - oy), min_lookahead_);
      steer = std::clamp(std::atan2(2.0 * wheelbase_ * std::sin(alpha), Ld),
                         -steer_max_, steer_max_);
    }
    cmd.linear.x = cruise_;
    cmd.angular.z = cruise_ * std::tan(steer) / wheelbase_;
    pub_->publish(cmd);

    // --- Debug (throttled) so you can see it working ---
    if (++log_ticks_ >= 40) {   // ~2 s
      log_ticks_ = 0;
      const double c = std::cos(spawn_yaw_), s = std::sin(spawn_yaw_);
      const double wx = spawn_x_ + c * ox - s * oy;
      const double wy = spawn_y_ + s * ox + c * oy;
      RCLCPP_INFO(this->get_logger(),
        "%s spoke#%d(%.0f deg) | dist_from_start=%.2f m | odom=(%.2f,%.2f) world=(%.2f,%.2f) | steer=%.2f",
        outbound_ ? "OUT " : "BACK", spoke_, spokeAngle(spoke_) * 180.0 / M_PI,
        d, ox, oy, wx, wy, steer);
    }
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  nav_msgs::msg::Odometry::SharedPtr odom_;
  bool have_odom_ = false;

  double cruise_, max_range_, min_range_, sweep_deg_, center_deg_;
  int num_spokes_;
  double wheelbase_, steer_max_, min_lookahead_;
  double spawn_x_, spawn_y_, spawn_yaw_;

  bool outbound_ = true;
  int spoke_ = 0;
  double theta_out_ = 0.0;
  int stuck_ticks_ = 0, log_ticks_ = 0;
  double last_x_ = 0.0, last_y_ = 0.0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AutoExplore>());
  rclcpp::shutdown();
  return 0;
}
