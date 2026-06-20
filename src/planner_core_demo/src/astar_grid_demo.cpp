#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
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

struct GridNode
{
  double f;
  int id;

  bool operator<(const GridNode & other) const
  {
    return f > other.f;
  }
};

class AStarGridDemoNode : public rclcpp::Node
{
public:
  AStarGridDemoNode() : Node("astar_grid_demo_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/planner_core_demo/markers", 10);

    initMapParams();
    generateObstacles();
    buildOccupancyGrid();
    runAStar();

    timer_ = this->create_wall_timer(
      500ms, std::bind(&AStarGridDemoNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "astar_grid_demo_node started.");
  }

private:
  void initMapParams()
  {
    resolution_ = 0.25;

    // Expanded arena: 16 x 14 x 4 m.
    x_min_ = -8.0;
    x_max_ = 8.0;
    y_min_ = -7.0;
    y_max_ = 7.0;
    z_min_ = 0.25;
    z_max_ = 4.25;

    nx_ = static_cast<int>((x_max_ - x_min_) / resolution_);
    ny_ = static_cast<int>((y_max_ - y_min_) / resolution_);
    nz_ = static_cast<int>((z_max_ - z_min_) / resolution_);

    start_x_ = -7.0;
    start_y_ = -6.0;
    start_z_ = 1.0;

    goal_x_ = 7.0;
    goal_y_ = 6.0;
    goal_z_ = 1.0;

    safety_margin_ = 0.15;

    // Scenario type.
    scenario_ = this->declare_parameter<std::string>("scenario", "dense");

    occupancy_.assign(nx_ * ny_ * nz_, 0);
  }

  void generateObstacles()
  {
    obstacles_.clear();
    const std::string scen = scenario_;
    std::mt19937 rng(7);

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
        if (std::hypot(obs.x - start_x_, obs.y - start_y_) < 1.8) continue;
        if (std::hypot(obs.x - goal_x_, obs.y - goal_y_) < 1.8) continue;
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
          if (std::hypot(obs.x - start_x_, obs.y - start_y_) < 1.8) continue;
          if (std::hypot(obs.x - goal_x_, obs.y - goal_y_) < 1.8) continue;
          obstacles_.push_back(obs);
        }
      }
    } else {
      std::uniform_real_distribution<double> pos_x(-7.2, 7.2), pos_y(-6.2, 6.2);
      std::uniform_real_distribution<double> size_xy(0.4, 0.85), height(0.9, 2.8);
      for (int i = 0; i < 55; ++i) {
        BoxObstacle obs;
        obs.x = pos_x(rng); obs.y = pos_y(rng);
        obs.sx = size_xy(rng); obs.sy = size_xy(rng); obs.sz = height(rng); obs.z = obs.sz / 2.0;
        if (std::hypot(obs.x - start_x_, obs.y - start_y_) < 1.8) continue;
        if (std::hypot(obs.x - goal_x_, obs.y - goal_y_) < 1.8) continue;
        obstacles_.push_back(obs);
      }
    }
    obstacles_.push_back(BoxObstacle{-1.2, -0.5, 1.6, 1.0, 2.8, 3.2});
    obstacles_.push_back(BoxObstacle{ 1.5,  1.2, 1.5, 2.4, 1.0, 3.0});
    obstacles_.push_back(BoxObstacle{ 2.5, -1.8, 1.3, 1.0, 2.6, 2.6});
  }

  bool worldToGrid(double x, double y, double z, int & ix, int & iy, int & iz) const
  {
    if (x < x_min_ || x >= x_max_ ||
        y < y_min_ || y >= y_max_ ||
        z < z_min_ || z >= z_max_) {
      return false;
    }

    ix = static_cast<int>((x - x_min_) / resolution_);
    iy = static_cast<int>((y - y_min_) / resolution_);
    iz = static_cast<int>((z - z_min_) / resolution_);

    return isValidIndex(ix, iy, iz);
  }

  geometry_msgs::msg::Point gridToWorld(int ix, int iy, int iz) const
  {
    geometry_msgs::msg::Point p;
    p.x = x_min_ + (ix + 0.5) * resolution_;
    p.y = y_min_ + (iy + 0.5) * resolution_;
    p.z = z_min_ + (iz + 0.5) * resolution_;
    return p;
  }

  bool isValidIndex(int ix, int iy, int iz) const
  {
    return ix >= 0 && ix < nx_ &&
           iy >= 0 && iy < ny_ &&
           iz >= 0 && iz < nz_;
  }

  int toAddress(int ix, int iy, int iz) const
  {
    return ix + nx_ * (iy + ny_ * iz);
  }

  void fromAddress(int id, int & ix, int & iy, int & iz) const
  {
    iz = id / (nx_ * ny_);
    int rest = id - iz * nx_ * ny_;
    iy = rest / nx_;
    ix = rest - iy * nx_;
  }

  bool isInsideObstacle(double x, double y, double z) const
  {
    for (const auto & obs : obstacles_) {
      if (std::abs(x - obs.x) <= obs.sx / 2.0 + safety_margin_ &&
          std::abs(y - obs.y) <= obs.sy / 2.0 + safety_margin_ &&
          std::abs(z - obs.z) <= obs.sz / 2.0 + safety_margin_) {
        return true;
      }
    }
    return false;
  }

  void buildOccupancyGrid()
  {
    std::fill(occupancy_.begin(), occupancy_.end(), 0);

    for (int iz = 0; iz < nz_; ++iz) {
      for (int iy = 0; iy < ny_; ++iy) {
        for (int ix = 0; ix < nx_; ++ix) {
          auto p = gridToWorld(ix, iy, iz);
          if (isInsideObstacle(p.x, p.y, p.z)) {
            occupancy_[toAddress(ix, iy, iz)] = 1;
          }
        }
      }
    }
  }

  bool isOccupied(int ix, int iy, int iz) const
  {
    if (!isValidIndex(ix, iy, iz)) {
      return true;
    }

    return occupancy_[toAddress(ix, iy, iz)] == 1;
  }

  double heuristic(int id, int goal_id) const
  {
    int ix, iy, iz;
    int gx, gy, gz;

    fromAddress(id, ix, iy, iz);
    fromAddress(goal_id, gx, gy, gz);

    double dx = static_cast<double>(ix - gx) * resolution_;
    double dy = static_cast<double>(iy - gy) * resolution_;
    double dz = static_cast<double>(iz - gz) * resolution_;

    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  bool runAStar()
  {
    int sx, sy, sz;
    int gx, gy, gz;

    if (!worldToGrid(start_x_, start_y_, start_z_, sx, sy, sz)) {
      RCLCPP_ERROR(this->get_logger(), "Start is outside map.");
      return false;
    }

    if (!worldToGrid(goal_x_, goal_y_, goal_z_, gx, gy, gz)) {
      RCLCPP_ERROR(this->get_logger(), "Goal is outside map.");
      return false;
    }

    int start_id = toAddress(sx, sy, sz);
    int goal_id = toAddress(gx, gy, gz);

    if (occupancy_[start_id] == 1) {
      RCLCPP_ERROR(this->get_logger(), "Start is occupied.");
      return false;
    }

    if (occupancy_[goal_id] == 1) {
      RCLCPP_ERROR(this->get_logger(), "Goal is occupied.");
      return false;
    }

    const int total_size = nx_ * ny_ * nz_;
    std::vector<double> g_score(total_size, std::numeric_limits<double>::infinity());
    std::vector<int> parent(total_size, -1);
    std::vector<int> closed(total_size, 0);

    std::priority_queue<GridNode> open_set;

    g_score[start_id] = 0.0;
    open_set.push(GridNode{heuristic(start_id, goal_id), start_id});

    bool success = false;

    const int dirs[26][3] = {
      {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1},
      {-1,  0, -1}, {-1,  0, 0}, {-1,  0, 1},
      {-1,  1, -1}, {-1,  1, 0}, {-1,  1, 1},
      { 0, -1, -1}, { 0, -1, 0}, { 0, -1, 1},
      { 0,  0, -1},               { 0,  0, 1},
      { 0,  1, -1}, { 0,  1, 0}, { 0,  1, 1},
      { 1, -1, -1}, { 1, -1, 0}, { 1, -1, 1},
      { 1,  0, -1}, { 1,  0, 0}, { 1,  0, 1},
      { 1,  1, -1}, { 1,  1, 0}, { 1,  1, 1}
    };

    while (!open_set.empty()) {
      auto current = open_set.top();
      open_set.pop();

      int current_id = current.id;

      if (closed[current_id]) {
        continue;
      }

      closed[current_id] = 1;

      if (current_id == goal_id) {
        success = true;
        break;
      }

      int cx, cy, cz;
      fromAddress(current_id, cx, cy, cz);

      for (const auto & dir : dirs) {
        int nx = cx + dir[0];
        int ny = cy + dir[1];
        int nz = cz + dir[2];

        if (!isValidIndex(nx, ny, nz)) {
          continue;
        }

        if (isOccupied(nx, ny, nz)) {
          continue;
        }

        int neighbor_id = toAddress(nx, ny, nz);

        if (closed[neighbor_id]) {
          continue;
        }

        double step_cost = resolution_ * std::sqrt(
          static_cast<double>(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]));

        double tentative_g = g_score[current_id] + step_cost;

        if (tentative_g < g_score[neighbor_id]) {
          g_score[neighbor_id] = tentative_g;
          parent[neighbor_id] = current_id;
          double f = tentative_g + heuristic(neighbor_id, goal_id);
          open_set.push(GridNode{f, neighbor_id});
        }
      }
    }

    path_.clear();

    if (!success) {
      RCLCPP_ERROR(this->get_logger(), "A* failed to find a path.");
      return false;
    }

    int id = goal_id;
    while (id != -1) {
      int ix, iy, iz;
      fromAddress(id, ix, iy, iz);
      path_.push_back(gridToWorld(ix, iy, iz));
      id = parent[id];
    }

    std::reverse(path_.begin(), path_.end());

    RCLCPP_INFO(
      this->get_logger(),
      "A* success. Path points: %zu, path length: %.2f m",
      path_.size(),
      computePathLength(path_));

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

    start.pose.position.x = start_x_;
    start.pose.position.y = start_y_;
    start.pose.position.z = start_z_;

    start.scale.x = 0.35;
    start.scale.y = 0.35;
    start.scale.z = 0.35;

    start.color.r = 0.0;
    start.color.g = 1.0;
    start.color.b = 0.0;
    start.color.a = 1.0;

    visualization_msgs::msg::Marker goal = start;
    goal.id = 1001;
    goal.pose.position.x = goal_x_;
    goal.pose.position.y = goal_y_;
    goal.pose.position.z = goal_z_;

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
    path.ns = "astar_path";
    path.id = 2000;
    path.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path.action = visualization_msgs::msg::Marker::ADD;

    path.scale.x = 0.07;

    path.color.r = 0.1;
    path.color.g = 0.9;
    path.color.b = 0.1;
    path.color.a = 1.0;

    path.points = path_;

    array.markers.push_back(path);
  }

  double resolution_;
  double x_min_;
  double x_max_;
  double y_min_;
  double y_max_;
  double z_min_;
  double z_max_;
  double safety_margin_;

  int nx_;
  int ny_;
  int nz_;

  std::string scenario_;

  double start_x_;
  double start_y_;
  double start_z_;
  double goal_x_;
  double goal_y_;
  double goal_z_;

  std::vector<int8_t> occupancy_;
  std::vector<BoxObstacle> obstacles_;
  std::vector<geometry_msgs::msg::Point> path_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AStarGridDemoNode>());
  rclcpp::shutdown();
  return 0;
}
