// planner_benchmark.cpp
//
// Task 15-⑤: Performance evaluation of the planning framework.
//
// Runs the full pipeline (Kinodynamic A* front-end + B-spline back-end) over a
// batch of randomised maps and reports the core metrics requested by the
// assignment: planning time, trajectory length and obstacle-avoidance success
// rate, plus smoothness, clearance and the EGO-vs-ESDF speed-up.
//
// The results are printed to the console AND written to a CSV file so they can be
// pasted into the evaluation report.  This is a head-less batch node: it does its
// work in the constructor and then asks rclcpp to shut down.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"

struct Vec3 { double x; double y; double z; };
struct BoxObstacle { double x; double y; double z; double sx; double sy; double sz; };

struct KinoNode
{
  Vec3 pos; Vec3 vel; double time;
  double g_score; double f_score; int parent; long long key;
};
struct QueueItem
{
  double f_score; int node_id;
  bool operator<(const QueueItem & o) const { return f_score > o.f_score; }
};

// One row of the benchmark table.
struct TrialResult
{
  bool frontend_success = false;
  double frontend_ms = 0.0;
  double esdf_total_ms = 0.0;
  double ego_total_ms = 0.0;
  double esdf_build_ms = 0.0;
  double length = 0.0;
  double smoothness = 0.0;
  double min_clearance = 0.0;
  bool collision_free = false;     // obstacle-avoidance success
};

class PlannerBenchmarkNode : public rclcpp::Node
{
public:
  PlannerBenchmarkNode() : Node("planner_benchmark_node")
  {
    num_trials_ = this->declare_parameter<int>("num_trials", 30);
    output_csv_ = this->declare_parameter<std::string>(
      "output_csv", "benchmark_results.csv");

    initParams();
    runBenchmark();

    // Batch job done -> shut the executor down so the process exits.
    rclcpp::shutdown();
  }

private:
  void initParams()
  {
    x_min_ = -8.0; x_max_ = 8.0; y_min_ = -7.0; y_max_ = 7.0; z_min_ = 0.25; z_max_ = 4.25;
    pos_resolution_ = 0.4; vel_resolution_ = 0.5;
    nx_ = static_cast<int>((x_max_ - x_min_) / pos_resolution_);
    ny_ = static_cast<int>((y_max_ - y_min_) / pos_resolution_);
    nz_ = static_cast<int>((z_max_ - z_min_) / pos_resolution_);
    max_vel_ = 3.5; max_acc_ = 2.5; dt_ = 0.4;
    primitive_check_num_ = 8; goal_tolerance_ = 0.65; safety_margin_ = 0.18;
    max_search_time_ = 25.0; max_expand_num_ = 200000;
    start_pos_ = Vec3{-7.0, -6.0, 1.0}; start_vel_ = Vec3{0.0, 0.0, 0.0};
    goal_pos_ = Vec3{7.0, 6.0, 1.0};
    nv_ = static_cast<int>(2.0 * max_vel_ / vel_resolution_) + 1;

    opt_iterations_ = 300; opt_step_ = 0.06;
    lambda_smooth_ = 1.0; lambda_collision_ = 2.2; d_safe_ = 0.6;
    bspline_samples_per_segment_ = 12; esdf_resolution_ = 0.12;
  }

  // -------------------------------------------------------------- benchmark
  void runBenchmark()
  {
    std::vector<TrialResult> results;
    RCLCPP_INFO(this->get_logger(), "Running %d benchmark trials...", num_trials_);

    for (int trial = 0; trial < num_trials_; ++trial) {
      // Each trial uses a different obstacle layout (different RNG seed) but the
      // same start/goal, so the difficulty varies trial to trial.
      generateObstacles(100 + trial);
      TrialResult r = runOneTrial();
      results.push_back(r);
      RCLCPP_INFO(this->get_logger(),
        "trial %2d | fe:%s fe_ms:%.2f | esdf_ms:%.2f ego_ms:%.2f | len:%.2f clr:%.2f free:%s",
        trial, r.frontend_success ? "ok" : "FAIL", r.frontend_ms,
        r.esdf_total_ms, r.ego_total_ms, r.length, r.min_clearance,
        r.collision_free ? "yes" : "no");
    }

    summarize(results);
    writeCsv(results);
  }

  TrialResult runOneTrial()
  {
    TrialResult r;
    auto t0 = std::chrono::steady_clock::now();
    const bool ok = runKinoAStar();
    auto t1 = std::chrono::steady_clock::now();
    r.frontend_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.frontend_success = ok;
    if (!ok) return r;

    generateControlPointsFromKinoPath();

    // ESDF back-end (build + optimise).
    auto e0 = std::chrono::steady_clock::now();
    buildEsdf();
    auto e1 = std::chrono::steady_clock::now();
    r.esdf_build_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();
    auto esdf_cps = optimize(/*use_esdf=*/true);
    auto e2 = std::chrono::steady_clock::now();
    r.esdf_total_ms = std::chrono::duration<double, std::milli>(e2 - e0).count();

    // EGO back-end (optimise only, no field).
    auto g0 = std::chrono::steady_clock::now();
    auto ego_cps = optimize(/*use_esdf=*/false);
    auto g1 = std::chrono::steady_clock::now();
    r.ego_total_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    // Quality metrics are reported from the EGO trajectory (the deployed planner).
    auto path = sampleBspline(ego_cps);
    r.length = computePathLength(path);
    r.smoothness = computeSmoothnessCost(path);
    r.min_clearance = computeMinClearance(path);
    r.collision_free = checkCollisionFree(path);
    (void)esdf_cps;
    return r;
  }

  void summarize(const std::vector<TrialResult> & results)
  {
    const int n = static_cast<int>(results.size());
    int fe_ok = 0, avoid_ok = 0;
    double sum_fe = 0, sum_esdf = 0, sum_ego = 0, sum_len = 0, sum_sm = 0, sum_clr = 0, sum_speedup = 0;
    int quality_n = 0;
    for (const auto & r : results) {
      if (r.frontend_success) {
        fe_ok++;
        sum_fe += r.frontend_ms; sum_esdf += r.esdf_total_ms; sum_ego += r.ego_total_ms;
        sum_len += r.length; sum_sm += r.smoothness; sum_clr += r.min_clearance;
        if (r.ego_total_ms > 1e-6) sum_speedup += r.esdf_total_ms / r.ego_total_ms;
        if (r.collision_free) avoid_ok++;
        quality_n++;
      }
    }
    const double q = std::max(1, quality_n);
    RCLCPP_INFO(this->get_logger(), "==================== BENCHMARK SUMMARY ====================");
    RCLCPP_INFO(this->get_logger(), "trials: %d", n);
    RCLCPP_INFO(this->get_logger(), "front-end success rate : %.1f%% (%d/%d)",
      100.0 * fe_ok / std::max(1, n), fe_ok, n);
    RCLCPP_INFO(this->get_logger(), "obstacle-avoidance rate: %.1f%% (%d/%d planned)",
      100.0 * avoid_ok / q, avoid_ok, quality_n);
    RCLCPP_INFO(this->get_logger(), "avg front-end time     : %.3f ms", sum_fe / q);
    RCLCPP_INFO(this->get_logger(), "avg ESDF back-end time : %.3f ms", sum_esdf / q);
    RCLCPP_INFO(this->get_logger(), "avg EGO  back-end time : %.3f ms", sum_ego / q);
    RCLCPP_INFO(this->get_logger(), "avg EGO total speed-up : %.2fx", sum_speedup / q);
    RCLCPP_INFO(this->get_logger(), "avg trajectory length  : %.3f m", sum_len / q);
    RCLCPP_INFO(this->get_logger(), "avg smoothness cost    : %.4f", sum_sm / q);
    RCLCPP_INFO(this->get_logger(), "avg min clearance      : %.3f m", sum_clr / q);
    RCLCPP_INFO(this->get_logger(), "===========================================================");
  }

  void writeCsv(const std::vector<TrialResult> & results)
  {
    std::ofstream f(output_csv_);
    if (!f.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Cannot open CSV file: %s", output_csv_.c_str());
      return;
    }
    f << "trial,frontend_success,frontend_ms,esdf_build_ms,esdf_total_ms,ego_total_ms,"
         "speedup,length_m,smoothness,min_clearance_m,collision_free\n";
    int i = 0;
    for (const auto & r : results) {
      const double sp = (r.ego_total_ms > 1e-6) ? r.esdf_total_ms / r.ego_total_ms : 0.0;
      f << i++ << ',' << (r.frontend_success ? 1 : 0) << ',' << r.frontend_ms << ','
        << r.esdf_build_ms << ',' << r.esdf_total_ms << ',' << r.ego_total_ms << ','
        << sp << ',' << r.length << ',' << r.smoothness << ',' << r.min_clearance << ','
        << (r.collision_free ? 1 : 0) << '\n';
    }
    f.close();
    RCLCPP_INFO(this->get_logger(), "CSV written to: %s", output_csv_.c_str());
  }

  // -------------------------------------------------------------- scene gen
  void generateObstacles(unsigned seed)
  {
    obstacles_.clear();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos_x(-7.2, 7.2);
    std::uniform_real_distribution<double> pos_y(-6.2, 6.2);
    std::uniform_real_distribution<double> size_xy(0.4, 0.85);
    std::uniform_real_distribution<double> height(0.9, 2.8);
    std::uniform_int_distribution<int> count(40, 55);
    const int n = count(rng);
    for (int i = 0; i < n; ++i) {
      BoxObstacle obs;
      obs.x = pos_x(rng); obs.y = pos_y(rng);
      obs.sx = size_xy(rng); obs.sy = size_xy(rng);
      obs.sz = height(rng); obs.z = obs.sz / 2.0;
      if (distance2D(obs.x, obs.y, start_pos_.x, start_pos_.y) < 1.8) continue;
      if (distance2D(obs.x, obs.y, goal_pos_.x, goal_pos_.y) < 1.8) continue;
      obstacles_.push_back(obs);
    }
  }

  // -------------------------------------------------------------- helpers
  double distance2D(double x1, double y1, double x2, double y2) const
  { const double dx = x1 - x2, dy = y1 - y2; return std::sqrt(dx * dx + dy * dy); }
  double norm(const Vec3 & v) const { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
  double dist(const Vec3 & a, const Vec3 & b) const
  { const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z; return std::sqrt(dx * dx + dy * dy + dz * dz); }
  bool isInsideMap(const Vec3 & p) const
  { return p.x >= x_min_ && p.x < x_max_ && p.y >= y_min_ && p.y < y_max_ && p.z >= z_min_ && p.z < z_max_; }
  bool isInsideObstacle(const Vec3 & p) const
  {
    for (const auto & obs : obstacles_)
      if (std::abs(p.x - obs.x) <= obs.sx / 2.0 + safety_margin_ &&
          std::abs(p.y - obs.y) <= obs.sy / 2.0 + safety_margin_ &&
          std::abs(p.z - obs.z) <= obs.sz / 2.0 + safety_margin_) return true;
    return false;
  }
  bool isStateValid(const Vec3 & p, const Vec3 & v) const
  { return isInsideMap(p) && !isInsideObstacle(p) && norm(v) <= max_vel_ + 1e-6; }
  double computeClearanceToBox(const Vec3 & p, const BoxObstacle & obs) const
  {
    const double dx = std::max(std::abs(p.x - obs.x) - obs.sx / 2.0, 0.0);
    const double dy = std::max(std::abs(p.y - obs.y) - obs.sy / 2.0, 0.0);
    const double dz = std::max(std::abs(p.z - obs.z) - obs.sz / 2.0, 0.0);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  // -------------------------------------------------------------- front end
  long long stateKey(const Vec3 & p, const Vec3 & v) const
  {
    int ix = static_cast<int>((p.x - x_min_) / pos_resolution_);
    int iy = static_cast<int>((p.y - y_min_) / pos_resolution_);
    int iz = static_cast<int>((p.z - z_min_) / pos_resolution_);
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) return -1;
    int ivx = static_cast<int>(std::round((v.x + max_vel_) / vel_resolution_));
    int ivy = static_cast<int>(std::round((v.y + max_vel_) / vel_resolution_));
    int ivz = static_cast<int>(std::round((v.z + max_vel_) / vel_resolution_));
    if (ivx < 0 || ivx >= nv_ || ivy < 0 || ivy >= nv_ || ivz < 0 || ivz >= nv_) return -1;
    long long key = ix; key = key * ny_ + iy; key = key * nz_ + iz;
    key = key * nv_ + ivx; key = key * nv_ + ivy; key = key * nv_ + ivz;
    return key;
  }
  Vec3 propagatePos(const Vec3 & p, const Vec3 & v, const Vec3 & a, double t) const
  { return Vec3{p.x + v.x * t + 0.5 * a.x * t * t, p.y + v.y * t + 0.5 * a.y * t * t, p.z + v.z * t + 0.5 * a.z * t * t}; }
  Vec3 propagateVel(const Vec3 & v, const Vec3 & a, double t) const
  { return Vec3{v.x + a.x * t, v.y + a.y * t, v.z + a.z * t}; }
  bool checkPrimitiveCollision(const Vec3 & p, const Vec3 & v, const Vec3 & a) const
  {
    for (int i = 1; i <= primitive_check_num_; ++i) {
      const double t = dt_ * static_cast<double>(i) / primitive_check_num_;
      if (!isStateValid(propagatePos(p, v, a, t), propagateVel(v, a, t))) return false;
    }
    return true;
  }
  std::vector<Vec3> generateControlSamples() const
  {
    std::vector<Vec3> c;
    const std::array<double, 3> values = {-max_acc_, 0.0, max_acc_};
    for (double ax : values) for (double ay : values) for (double az : values) {
      Vec3 a{ax, ay, az}; if (norm(a) <= max_acc_ * 1.75) c.push_back(a);
    }
    return c;
  }
  double heuristic(const Vec3 & p, const Vec3 & v) const
  {
    const double d = dist(p, goal_pos_);
    Vec3 tg{goal_pos_.x - p.x, goal_pos_.y - p.y, goal_pos_.z - p.z};
    const double dn = std::max(d, 1e-3);
    Vec3 dv{max_vel_ * tg.x / dn, max_vel_ * tg.y / dn, max_vel_ * tg.z / dn};
    return d / max_vel_ + 0.08 * dist(v, dv);
  }
  bool runKinoAStar()
  {
    if (!isStateValid(start_pos_, start_vel_)) { kino_path_.clear(); return false; }
    const auto controls = generateControlSamples();
    std::vector<KinoNode> nodes;
    std::priority_queue<QueueItem> open_set;
    std::unordered_map<long long, double> best_g;
    std::unordered_map<long long, int> best_id;
    std::unordered_set<long long> closed;
    long long sk = stateKey(start_pos_, start_vel_);
    KinoNode s; s.pos = start_pos_; s.vel = start_vel_; s.time = 0.0;
    s.g_score = 0.0; s.f_score = heuristic(start_pos_, start_vel_); s.parent = -1; s.key = sk;
    nodes.push_back(s); best_g[sk] = 0.0; best_id[sk] = 0; open_set.push(QueueItem{s.f_score, 0});
    int final_id = -1, expand = 0;
    while (!open_set.empty() && expand < max_expand_num_) {
      QueueItem it = open_set.top(); open_set.pop();
      const KinoNode cur = nodes[it.node_id];
      auto bit = best_id.find(cur.key);
      if (bit == best_id.end() || bit->second != it.node_id) continue;
      if (closed.count(cur.key)) continue;
      closed.insert(cur.key); expand++;
      if (dist(cur.pos, goal_pos_) < goal_tolerance_ && cur.time > 1.0) { final_id = it.node_id; break; }
      if (cur.time > max_search_time_) continue;
      for (const auto & acc : controls) {
        if (!checkPrimitiveCollision(cur.pos, cur.vel, acc)) continue;
        Vec3 np = propagatePos(cur.pos, cur.vel, acc, dt_);
        Vec3 nv = propagateVel(cur.vel, acc, dt_);
        long long nk = stateKey(np, nv);
        if (nk < 0 || closed.count(nk)) continue;
        const double ng = cur.g_score + dt_ + 0.03 * norm(acc) * norm(acc) * dt_;
        auto git = best_g.find(nk);
        if (git == best_g.end() || ng < git->second) {
          KinoNode n; n.pos = np; n.vel = nv; n.time = cur.time + dt_;
          n.g_score = ng; n.f_score = ng + heuristic(np, nv); n.parent = it.node_id; n.key = nk;
          int id = static_cast<int>(nodes.size()); nodes.push_back(n);
          best_g[nk] = ng; best_id[nk] = id; open_set.push(QueueItem{n.f_score, id});
        }
      }
    }
    kino_path_.clear();
    if (final_id < 0) return false;
    std::vector<int> ids; int id = final_id;
    while (id >= 0) { ids.push_back(id); id = nodes[id].parent; }
    std::reverse(ids.begin(), ids.end());
    for (int nid : ids) {
      geometry_msgs::msg::Point p;
      p.x = nodes[nid].pos.x; p.y = nodes[nid].pos.y; p.z = nodes[nid].pos.z;
      kino_path_.push_back(p);
    }
    return true;
  }

  // -------------------------------------------------------------- back ends
  void generateControlPointsFromKinoPath()
  {
    cps_.clear();
    if (kino_path_.size() < 2) return;
    cps_.push_back(kino_path_.front()); cps_.push_back(kino_path_.front());
    for (const auto & p : kino_path_) cps_.push_back(p);
    cps_.push_back(kino_path_.back()); cps_.push_back(kino_path_.back());
  }
  geometry_msgs::msg::Point cubicBsplinePoint(
    const geometry_msgs::msg::Point & p0, const geometry_msgs::msg::Point & p1,
    const geometry_msgs::msg::Point & p2, const geometry_msgs::msg::Point & p3, double u) const
  {
    const double u2 = u * u, u3 = u2 * u;
    const double b0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0;
    const double b1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0;
    const double b2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0;
    const double b3 = u3 / 6.0;
    geometry_msgs::msg::Point p;
    p.x = b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x;
    p.y = b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y;
    p.z = b0 * p0.z + b1 * p1.z + b2 * p2.z + b3 * p3.z;
    return p;
  }
  std::vector<geometry_msgs::msg::Point> sampleBspline(
    const std::vector<geometry_msgs::msg::Point> & cps) const
  {
    std::vector<geometry_msgs::msg::Point> path;
    if (cps.size() < 4) return path;
    for (size_t i = 0; i + 3 < cps.size(); ++i)
      for (int j = 0; j < bspline_samples_per_segment_; ++j) {
        const double u = static_cast<double>(j) / bspline_samples_per_segment_;
        path.push_back(cubicBsplinePoint(cps[i], cps[i + 1], cps[i + 2], cps[i + 3], u));
      }
    path.push_back(cps.back());
    return path;
  }
  void addSmoothnessGradient(const std::vector<geometry_msgs::msg::Point> & q,
                             std::vector<Vec3> & grad) const
  {
    for (size_t i = 1; i + 1 < q.size(); ++i) {
      Vec3 a{q[i + 1].x - 2.0 * q[i].x + q[i - 1].x,
             q[i + 1].y - 2.0 * q[i].y + q[i - 1].y,
             q[i + 1].z - 2.0 * q[i].z + q[i - 1].z};
      grad[i - 1].x += lambda_smooth_ * 2.0 * a.x; grad[i - 1].y += lambda_smooth_ * 2.0 * a.y; grad[i - 1].z += lambda_smooth_ * 2.0 * a.z;
      grad[i].x     -= lambda_smooth_ * 4.0 * a.x; grad[i].y     -= lambda_smooth_ * 4.0 * a.y; grad[i].z     -= lambda_smooth_ * 4.0 * a.z;
      grad[i + 1].x += lambda_smooth_ * 2.0 * a.x; grad[i + 1].y += lambda_smooth_ * 2.0 * a.y; grad[i + 1].z += lambda_smooth_ * 2.0 * a.z;
    }
  }
  // ESDF grid
  void buildEsdf()
  {
    esdf_nx_ = static_cast<int>((x_max_ - x_min_) / esdf_resolution_) + 1;
    esdf_ny_ = static_cast<int>((y_max_ - y_min_) / esdf_resolution_) + 1;
    esdf_nz_ = static_cast<int>((z_max_ - z_min_) / esdf_resolution_) + 1;
    esdf_.assign(static_cast<size_t>(esdf_nx_) * esdf_ny_ * esdf_nz_, 0.0);
    for (int ix = 0; ix < esdf_nx_; ++ix) {
      const double x = x_min_ + ix * esdf_resolution_;
      for (int iy = 0; iy < esdf_ny_; ++iy) {
        const double y = y_min_ + iy * esdf_resolution_;
        for (int iz = 0; iz < esdf_nz_; ++iz) {
          const double z = z_min_ + iz * esdf_resolution_;
          const Vec3 p{x, y, z};
          double best = std::numeric_limits<double>::infinity();
          for (const auto & obs : obstacles_) best = std::min(best, computeClearanceToBox(p, obs));
          esdf_[(static_cast<size_t>(ix) * esdf_ny_ + iy) * esdf_nz_ + iz] = best;
        }
      }
    }
  }
  double esdfAt(const Vec3 & p) const
  {
    int ix = std::clamp(static_cast<int>((p.x - x_min_) / esdf_resolution_), 0, esdf_nx_ - 1);
    int iy = std::clamp(static_cast<int>((p.y - y_min_) / esdf_resolution_), 0, esdf_ny_ - 1);
    int iz = std::clamp(static_cast<int>((p.z - z_min_) / esdf_resolution_), 0, esdf_nz_ - 1);
    return esdf_[(static_cast<size_t>(ix) * esdf_ny_ + iy) * esdf_nz_ + iz];
  }
  Vec3 esdfCollisionGradient(const geometry_msgs::msg::Point & pt) const
  {
    const Vec3 p{pt.x, pt.y, pt.z};
    const double d = esdfAt(p);
    if (d >= d_safe_) return Vec3{0.0, 0.0, 0.0};
    const double h = esdf_resolution_;
    Vec3 g{(esdfAt(Vec3{p.x + h, p.y, p.z}) - esdfAt(Vec3{p.x - h, p.y, p.z})) / (2.0 * h),
           (esdfAt(Vec3{p.x, p.y + h, p.z}) - esdfAt(Vec3{p.x, p.y - h, p.z})) / (2.0 * h),
           (esdfAt(Vec3{p.x, p.y, p.z + h}) - esdfAt(Vec3{p.x, p.y, p.z - h})) / (2.0 * h)};
    const double coef = -2.0 * lambda_collision_ * (d_safe_ - d);
    return Vec3{coef * g.x, coef * g.y, coef * g.z};
  }
  Vec3 egoCollisionGradient(const geometry_msgs::msg::Point & pt) const
  {
    const Vec3 q{pt.x, pt.y, pt.z};
    double best = std::numeric_limits<double>::infinity();
    const BoxObstacle * bo = nullptr;
    for (const auto & obs : obstacles_) { const double c = computeClearanceToBox(q, obs); if (c < best) { best = c; bo = &obs; } }
    if (bo == nullptr || best >= d_safe_) return Vec3{0.0, 0.0, 0.0};
    const double hx = bo->sx / 2.0, hy = bo->sy / 2.0, hz = bo->sz / 2.0;
    const double cx = std::clamp(q.x, bo->x - hx, bo->x + hx);
    const double cy = std::clamp(q.y, bo->y - hy, bo->y + hy);
    const double cz = std::clamp(q.z, bo->z - hz, bo->z + hz);
    Vec3 v{q.x - cx, q.y - cy, q.z - cz};
    double n = norm(v);
    if (n < 1e-6) {
      const double dxf = hx - std::abs(q.x - bo->x), dyf = hy - std::abs(q.y - bo->y), dzf = hz - std::abs(q.z - bo->z);
      if (dxf <= dyf && dxf <= dzf) v = Vec3{(q.x >= bo->x) ? 1.0 : -1.0, 0, 0};
      else if (dyf <= dzf)         v = Vec3{0, (q.y >= bo->y) ? 1.0 : -1.0, 0};
      else                          v = Vec3{0, 0, (q.z >= bo->z) ? 1.0 : -1.0};
      n = 1.0;
    }
    v.x /= n; v.y /= n; v.z /= n;
    const double coef = -2.0 * lambda_collision_ * (d_safe_ - best);
    return Vec3{coef * v.x, coef * v.y, coef * v.z};
  }
  std::vector<geometry_msgs::msg::Point> optimize(bool use_esdf)
  {
    auto q = cps_;
    for (int iter = 0; iter < opt_iterations_; ++iter) {
      std::vector<Vec3> grad(q.size(), Vec3{0.0, 0.0, 0.0});
      addSmoothnessGradient(q, grad);
      for (size_t i = 0; i < q.size(); ++i) {
        Vec3 g = use_esdf ? esdfCollisionGradient(q[i]) : egoCollisionGradient(q[i]);
        grad[i].x += g.x; grad[i].y += g.y; grad[i].z += g.z;
      }
      for (size_t i = 2; i + 2 < q.size(); ++i) {
        q[i].x -= opt_step_ * grad[i].x; q[i].y -= opt_step_ * grad[i].y; q[i].z -= opt_step_ * grad[i].z;
        q[i].x = std::clamp(q[i].x, x_min_ + 0.2, x_max_ - 0.2);
        q[i].y = std::clamp(q[i].y, y_min_ + 0.2, y_max_ - 0.2);
        q[i].z = std::clamp(q[i].z, z_min_ + 0.2, z_max_ - 0.2);
      }
    }
    return q;
  }

  // -------------------------------------------------------------- metrics
  double computePathLength(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    double l = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
      const double dx = path[i].x - path[i - 1].x, dy = path[i].y - path[i - 1].y, dz = path[i].z - path[i - 1].z;
      l += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return l;
  }
  double computeSmoothnessCost(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    if (path.size() < 3) return 0.0;
    double c = 0.0;
    for (size_t i = 1; i + 1 < path.size(); ++i) {
      const double ax = path[i + 1].x - 2.0 * path[i].x + path[i - 1].x;
      const double ay = path[i + 1].y - 2.0 * path[i].y + path[i - 1].y;
      const double az = path[i + 1].z - 2.0 * path[i].z + path[i - 1].z;
      c += ax * ax + ay * ay + az * az;
    }
    return c;
  }
  double computeMinClearance(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    if (path.empty() || obstacles_.empty()) return 0.0;
    double m = std::numeric_limits<double>::infinity();
    for (const auto & pt : path) {
      const Vec3 p{pt.x, pt.y, pt.z};
      for (const auto & obs : obstacles_) m = std::min(m, computeClearanceToBox(p, obs));
    }
    return std::isfinite(m) ? m : 0.0;
  }
  bool checkCollisionFree(const std::vector<geometry_msgs::msg::Point> & path) const
  {
    for (const auto & pt : path) {
      const Vec3 p{pt.x, pt.y, pt.z};
      if (!isInsideMap(p) || isInsideObstacle(p)) return false;
    }
    return true;
  }

  // -------------------------------------------------------------- data
  int num_trials_;
  std::string output_csv_;
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  double pos_resolution_, vel_resolution_;
  int nx_, ny_, nz_, nv_;
  double max_vel_, max_acc_, dt_;
  int primitive_check_num_;
  double goal_tolerance_, safety_margin_, max_search_time_;
  int max_expand_num_;
  Vec3 start_pos_, start_vel_, goal_pos_;
  int opt_iterations_;
  double opt_step_, lambda_smooth_, lambda_collision_, d_safe_;
  int bspline_samples_per_segment_;
  double esdf_resolution_;
  std::vector<BoxObstacle> obstacles_;
  std::vector<geometry_msgs::msg::Point> kino_path_;
  std::vector<geometry_msgs::msg::Point> cps_;
  std::vector<double> esdf_;
  int esdf_nx_ = 0, esdf_ny_ = 0, esdf_nz_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PlannerBenchmarkNode>();
  rclcpp::shutdown();
  return 0;
}
