#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <thread>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive.hpp"

using namespace std::chrono_literals;

// ─── Terminal raw-mode helpers ──────────────────────────────────────
static struct termios g_original_termios;

static void enableRawMode()
{
  tcgetattr(STDIN_FILENO, &g_original_termios);
  struct termios raw = g_original_termios;
  raw.c_lflag &= ~(ICANON | ECHO);
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSANOW, &raw);
}

static void restoreTerminal()
{
  tcsetattr(STDIN_FILENO, TCSANOW, &g_original_termios);
}

// ─── Node ────────────────────────────────────────────────────────────
class AckermannTeleop : public rclcpp::Node
{
public:
  AckermannTeleop()
  : Node("ackermann_teleop"),
    linear_vel_(0.0),
    steering_angle_(0.0),
    running_(true)
  {
    pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDrive>(
      "/ackermann_steering_controller/reference", 10);

    timer_ = this->create_wall_timer(
      50ms, std::bind(&AckermannTeleop::timerCallback, this));

    RCLCPP_INFO(this->get_logger(),
      "Ackermann Teleop started.\n"
      "  W / ↑   : speed up\n"
      "  S / ↓   : slow down\n"
      "  A / ←   : steer left\n"
      "  D / →   : steer right\n"
      "  Space   : emergency stop\n"
      "  Q       : quit");
  }

  ~AckermannTeleop() override { cleanup(); }

  void spinKeyboard()
  {
    char c;
    while (running_ && rclcpp::ok()) {
      c = getchar();
      switch (c) {
        case 'w': case 'W':
          linear_vel_ = clamp(linear_vel_ + 0.2, 0.0, 3.0);
          break;
        case 's': case 'S':
          linear_vel_ = clamp(linear_vel_ - 0.2, 0.0, 3.0);
          break;
        case 'a': case 'A':
          steering_angle_ = clamp(steering_angle_ + 0.1, -0.5, 0.5);
          break;
        case 'd': case 'D':
          steering_angle_ = clamp(steering_angle_ - 0.1, -0.5, 0.5);
          break;
        case ' ':
          linear_vel_ = 0.0;
          steering_angle_ = 0.0;
          break;
        case 'q': case 'Q':
          running_ = false;
          break;
        case 27: {  // ESC — arrow keys send ESC [ A/B/C/D
          char c2 = getchar();
          if (c2 == '[') {
            char c3 = getchar();
            switch (c3) {
              case 'A':  // ↑
                linear_vel_ = clamp(linear_vel_ + 0.2, 0.0, 3.0);
                break;
              case 'B':  // ↓
                linear_vel_ = clamp(linear_vel_ - 0.2, 0.0, 3.0);
                break;
              case 'C':  // →
                steering_angle_ = clamp(steering_angle_ - 0.1, -0.5, 0.5);
                break;
              case 'D':  // ←
                steering_angle_ = clamp(steering_angle_ + 0.1, -0.5, 0.5);
                break;
            }
          }
          break;
        }
        default:
          break;
      }

      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        "speed = %.1f m/s  |  steer = %.2f rad",
        linear_vel_.load(), steering_angle_.load());
    }
  }

  bool isRunning() const { return running_; }

private:
  void timerCallback()
  {
    auto msg = ackermann_msgs::msg::AckermannDrive();
    msg.speed = linear_vel_;
    msg.steering_angle = steering_angle_;
    pub_->publish(msg);
  }

  void cleanup()
  {
    running_ = false;
    restoreTerminal();
    RCLCPP_INFO(this->get_logger(), "Ackermann Teleop terminated.");
  }

  static double clamp(double val, double lo, double hi)
  {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
  }

  rclcpp::Publisher<ackermann_msgs::msg::AckermannDrive>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::atomic<double> linear_vel_;
  std::atomic<double> steering_angle_;
  std::atomic<bool> running_;
};

// ─── main ────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
  enableRawMode();
  atexit(restoreTerminal);

  rclcpp::init(argc, argv);
  auto node = std::make_shared<AckermannTeleop>();

  // Keyboard input on a separate thread
  std::thread kb_thread(&AckermannTeleop::spinKeyboard, node);

  rclcpp::spin(node);

  if (kb_thread.joinable()) {
    kb_thread.join();
  }

  rclcpp::shutdown();
  return 0;
}