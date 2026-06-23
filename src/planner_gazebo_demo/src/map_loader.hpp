// map_loader.hpp
//
// Utility to load a PGM + YAML map pair and provide grid-based collision
// checking for the planner nodes.
//
// The PGM file is parsed manually (P5 binary format). The YAML file provides
// resolution, origin, and thresholds.

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "yaml-cpp/yaml.h"

struct MapInfo
{
  double resolution;
  double origin_x;
  double origin_y;
  double origin_yaw;
  int width;
  int height;
  int occupied_thresh;   // percentage 0-100
  int free_thresh;       // percentage 0-100
  std::string image;
};

class MapLoader
{
public:
  bool load(const std::string & yaml_path)
  {
    // Parse YAML
    YAML::Node config;
    try {
      config = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception & e) {
      return false;
    }

    info_.resolution = config["resolution"].as<double>();
    auto origin = config["origin"].as<std::vector<double>>();
    info_.origin_x = origin[0];
    info_.origin_y = origin[1];
    info_.origin_yaw = origin.size() > 2 ? origin[2] : 0.0;
    info_.occupied_thresh = config["occupied_thresh"].as<int>();
    info_.free_thresh = config["free_thresh"].as<int>();
    info_.image = config["image"].as<std::string>();

    // Resolve image path relative to YAML directory
    std::string dir = yaml_path.substr(0, yaml_path.find_last_of("/"));
    std::string pgm_path = dir + "/" + info_.image;

    if (!loadPgm(pgm_path)) {
      return false;
    }

    buildOccupancyGrid();
    return true;
  }

  const nav_msgs::msg::OccupancyGrid & getOccupancyGrid() const { return grid_; }
  const MapInfo & getInfo() const { return info_; }

  // World -> grid cell
  bool worldToGrid(double wx, double wy, int & gx, int & gy) const
  {
    gx = static_cast<int>((wx - info_.origin_x) / info_.resolution);
    gy = static_cast<int>((wy - info_.origin_y) / info_.resolution);
    return gx >= 0 && gx < info_.width && gy >= 0 && gy < info_.height;
  }

  // Grid cell -> world
  void gridToWorld(int gx, int gy, double & wx, double & wy) const
  {
    wx = info_.origin_x + (gx + 0.5) * info_.resolution;
    wy = info_.origin_y + (gy + 0.5) * info_.resolution;
  }

  // Check if a world point is in collision (occupied or unknown)
  bool isOccupied(double wx, double wy, double margin = 0.0) const
  {
    if (margin <= 1e-6) {
      int gx, gy;
      if (!worldToGrid(wx, wy, gx, gy)) return true;  // outside = obstacle
      int idx = gy * info_.width + gx;
      return grid_.data[idx] > 50;  // occupied
    }
    // Check with margin (check surrounding cells)
    int mr = static_cast<int>(std::ceil(margin / info_.resolution));
    int cx, cy;
    if (!worldToGrid(wx, wy, cx, cy)) return true;
    for (int dx = -mr; dx <= mr; ++dx) {
      for (int dy = -mr; dy <= mr; ++dy) {
        int gx = cx + dx, gy = cy + dy;
        if (gx < 0 || gx >= info_.width || gy < 0 || gy >= info_.height) return true;
        if (grid_.data[gy * info_.width + gx] > 50) {
          // Check if within margin distance
          double cell_wx, cell_wy;
          gridToWorld(gx, gy, cell_wx, cell_wy);
          if (std::hypot(cell_wx - wx, cell_wy - wy) <= margin) return true;
        }
      }
    }
    return false;
  }

  // Check if a world point is in bounds and free
  bool isFree(double wx, double wy, double margin = 0.0) const
  {
    int gx, gy;
    if (!worldToGrid(wx, wy, gx, gy)) return false;
    if (margin > 1e-6) return !isOccupied(wx, wy, margin);
    int idx = gy * info_.width + gx;
    return grid_.data[idx] >= 0 && grid_.data[idx] < 50;
  }

private:
  bool loadPgm(const std::string & path)
  {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    std::string magic;
    f >> magic;
    if (magic != "P5") return false;

    // Skip comments
    std::string line;
    while (f.peek() == '#') std::getline(f, line);

    f >> info_.width >> info_.height;
    int maxval;
    f >> maxval;
    f.ignore(1);  // single whitespace after maxval

    // Read pixel data
    raw_data_.resize(info_.width * info_.height);
    f.read(reinterpret_cast<char *>(raw_data_.data()), raw_data_.size());
    return f.good() || f.eof();
  }

  void buildOccupancyGrid()
  {
    grid_.header.frame_id = "map";
    grid_.info.resolution = info_.resolution;
    grid_.info.width = info_.width;
    grid_.info.height = info_.height;
    grid_.info.origin.position.x = info_.origin_x;
    grid_.info.origin.position.y = info_.origin_y;
    grid_.info.origin.position.z = 0.0;
    grid_.info.origin.orientation.w = 1.0;

    grid_.data.resize(info_.width * info_.height);
    for (size_t i = 0; i < raw_data_.size(); ++i) {
      // PGM: 0=black(occupied), 255=white(free)
      // OccupancyGrid: 0=free, 100=occupied, -1=unknown
      int val = static_cast<int>(raw_data_[i]);
      int occ;
      if (val <= 10) {
        occ = 100;
      } else if (val >= 250) {
        occ = 0;
      } else {
        // Linear interpolation
        occ = static_cast<int>((1.0 - val / 255.0) * 100.0);
        if (occ > 100) occ = 100;
        if (occ < 0) occ = 0;
      }
      grid_.data[i] = occ;
    }
  }

  MapInfo info_;
  std::vector<uint8_t> raw_data_;
  nav_msgs::msg::OccupancyGrid grid_;
};