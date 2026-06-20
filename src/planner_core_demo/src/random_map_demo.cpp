#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"

using namespace std::chrono_literals;

struct BoxObstacle
{
  double x;
  double y;
  double z;
  double sx;
  double sy;
  double sz;
};

class RandomMapDemoNode : public rclcpp::Node
{
public:
  RandomMapDemoNode() : Node("random_map_demo_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/markers", 10);

    generateObstacles();

    timer_ = this->create_wall_timer(
      500ms, std::bind(&RandomMapDemoNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "planner_core_demo started.");
  }

private:
  void generateObstacles()
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pos_x(-7.2, 7.2);
    std::uniform_real_distribution<double> pos_y(-6.2, 6.2);
    std::uniform_real_distribution<double> height(0.6, 2.8);
    std::uniform_real_distribution<double> size_xy(0.35, 0.9);

    obstacles_.clear();

    for (int i = 0; i < 50; ++i) {
      BoxObstacle obs;
      obs.x = pos_x(rng);
      obs.y = pos_y(rng);
      obs.z = height(rng) / 2.0;
      obs.sx = size_xy(rng);
      obs.sy = size_xy(rng);
      obs.sz = height(rng);

      if (std::hypot(obs.x + 7.0, obs.y + 6.0) < 1.8) {
        continue;
      }
      if (std::hypot(obs.x - 7.0, obs.y - 6.0) < 1.8) {
        continue;
      }

      obstacles_.push_back(obs);
    }
  }

  bool isInsideObstacle(double x, double y, double z) const
  {
    for (const auto & obs : obstacles_) {
      if (std::abs(x - obs.x) <= obs.sx / 2.0 &&
          std::abs(y - obs.y) <= obs.sy / 2.0 &&
          std::abs(z - obs.z) <= obs.sz / 2.0) {
        return true;
      }
    }
    return false;
  }

  std::vector<geometry_msgs::msg::Point> generateSimplePath() const
  {
    std::vector<geometry_msgs::msg::Point> path;

    for (int i = 0; i <= 120; ++i) {
      double t = static_cast<double>(i) / 120.0;

      geometry_msgs::msg::Point p;
      p.x = -5.0 + 10.0 * t;
      p.y = -4.0 + 8.0 * t + 1.0 * std::sin(2.0 * M_PI * t);
      p.z = 1.0 + 0.4 * std::sin(M_PI * t);

      if (isInsideObstacle(p.x, p.y, p.z)) {
        p.z += 1.2;
      }

      path.push_back(p);
    }

    return path;
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
      m.color.g = 0.2;
      m.color.b = 0.2;
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
    start.pose.position.x = -5.0;
    start.pose.position.y = -4.0;
    start.pose.position.z = 1.0;
    start.scale.x = 0.3;
    start.scale.y = 0.3;
    start.scale.z = 0.3;
    start.color.r = 0.0;
    start.color.g = 1.0;
    start.color.b = 0.0;
    start.color.a = 1.0;

    visualization_msgs::msg::Marker goal = start;
    goal.id = 1001;
    goal.pose.position.x = 5.0;
    goal.pose.position.y = 4.0;
    goal.pose.position.z = 1.0;
    goal.color.r = 0.0;
    goal.color.g = 0.3;
    goal.color.b = 1.0;
    goal.color.a = 1.0;

    array.markers.push_back(start);
    array.markers.push_back(goal);
  }

  void addPathMarker(visualization_msgs::msg::MarkerArray & array)
  {
    auto path_points = generateSimplePath();

    visualization_msgs::msg::Marker path;
    path.header.frame_id = "map";
    path.header.stamp = this->now();
    path.ns = "initial_path";
    path.id = 2000;
    path.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path.action = visualization_msgs::msg::Marker::ADD;

    path.scale.x = 0.06;

    path.color.r = 0.1;
    path.color.g = 0.9;
    path.color.b = 0.1;
    path.color.a = 1.0;

    path.points = path_points;

    array.markers.push_back(path);
  }

  std::vector<BoxObstacle> obstacles_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RandomMapDemoNode>());
  rclcpp::shutdown();
  return 0;
}
