// ackermann_teleop.cpp
//
// Keyboard teleop node for the Ackermann car.
// Publishes AckermannDriveStamped on /ackermann_cmd.
// Controls: w/s = speed up/down, a/d = steer left/right, space = stop.

#include <termios.h>
#include <unistd.h>

#include <cmath>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>

class AckermannTeleop : public rclcpp::Node
{
public:
  AckermannTeleop() : Node("ackermann_teleop")
  {
    pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "/ackermann_steering_controller/reference", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&AckermannTeleop::timerCallback, this));
    RCLCPP_INFO(this->get_logger(),
      "Teleop ready: w/s=speed, a/d=steer, space=stop, q=quit");
  }

  void spin()
  {
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    char c;
    while (rclcpp::ok()) {
      if (read(STDIN_FILENO, &c, 1) > 0) {
        switch (c) {
          case 'w': speed_ += 0.2; break;
          case 's': speed_ -= 0.2; break;
          case 'a': steering_ += 0.1; break;
          case 'd': steering_ -= 0.1; break;
          case ' ': speed_ = 0.0; steering_ = 0.0; break;
          case 'q': rclcpp::shutdown(); break;
        }
        speed_ = std::clamp(speed_, -2.0, 2.0);
        steering_ = std::clamp(steering_, -0.61, 0.61);
      }
      rclcpp::spin_some(shared_from_this());
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  }

private:
  void timerCallback()
  {
    ackermann_msgs::msg::AckermannDriveStamped msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "base_link";
    msg.drive.speed = speed_;
    msg.drive.steering_angle = steering_;
    pub_->publish(msg);
  }

  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  double speed_ = 0.0;
  double steering_ = 0.0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AckermannTeleop>();
  node->spin();
  rclcpp::shutdown();
  return 0;
}