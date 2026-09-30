#include "OrbSlamRunner.h"

#include <Tracking.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

namespace {
constexpr double SLOW_TRACKING_THRESHOLD_MS = 100.0;
constexpr double SIGNIFICANT_LOSS_THRESHOLD_S = 0.1;
constexpr std::size_t SIGNIFICANT_LOSS_MIN_FRAMES = 3;
}

OrbSlamRunner::OrbSlamRunner(LiveSensorMode sensor,
                             const std::string& vocabulary_path,
                             const std::string& settings_path,
                             bool use_viewer,
                             const std::string& output_dir)
    : sensor_(sensor), output_dir_(output_dir) {
    makeDirectories(output_dir_);

    stats_file_.open(joinPath(output_dir_, "live_tracking_stats.csv").c_str(),
                     std::ios::out | std::ios::trunc);
    if (!stats_file_) {
        throw std::runtime_error("Could not create live_tracking_stats.csv in " + output_dir_);
    }

    stats_file_
        << "frame,timestamp_s,frame_dt_ms,state,track_ms,slow_track,"
        << "keypoints,map_point_associations\n";

    slam_.reset(new ORB_SLAM3::System(
        vocabulary_path,
        settings_path,
        toOrbSensor(sensor_),
        use_viewer));

    tracking_times_ms_.reserve(10000);

    std::cout
        << "ORB-SLAM3 started in " << LiveSensorModeName(sensor_) << " mode.\n"
        << "Statistics: " << joinPath(output_dir_, "live_tracking_stats.csv") << '\n';
}

OrbSlamRunner::~OrbSlamRunner() {
    try {
        finish();
    } catch (...) {
    }
}

ORB_SLAM3::System::eSensor OrbSlamRunner::toOrbSensor(LiveSensorMode sensor) {
    switch (sensor) {
        case LiveSensorMode::MONO:
            return ORB_SLAM3::System::MONOCULAR;
        case LiveSensorMode::MONO_IMU:
            return ORB_SLAM3::System::IMU_MONOCULAR;
        case LiveSensorMode::STEREO:
            return ORB_SLAM3::System::STEREO;
        case LiveSensorMode::STEREO_IMU:
            return ORB_SLAM3::System::IMU_STEREO;
        case LiveSensorMode::RGBD:
            return ORB_SLAM3::System::RGBD;
        default:
            throw std::runtime_error("Unsupported ORB-SLAM sensor mode");
    }
}

const char* OrbSlamRunner::trackingStateName(int state) {
    switch (state) {
        case ORB_SLAM3::Tracking::SYSTEM_NOT_READY: return "SYSTEM_NOT_READY";
        case ORB_SLAM3::Tracking::NO_IMAGES_YET:    return "NO_IMAGES_YET";
        case ORB_SLAM3::Tracking::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case ORB_SLAM3::Tracking::OK:              return "OK";
        case ORB_SLAM3::Tracking::RECENTLY_LOST:   return "RECENTLY_LOST";
        case ORB_SLAM3::Tracking::LOST:            return "LOST";
        case ORB_SLAM3::Tracking::OK_KLT:          return "OK_KLT";
        default:                                    return "UNKNOWN";
    }
}

double OrbSlamRunner::seconds(std::int64_t timestamp_ns) {
    return static_cast<double>(timestamp_ns) * 1e-9;
}

std::string OrbSlamRunner::joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a[a.size() - 1] == '/') return a + b;
    return a + "/" + b;
}

void OrbSlamRunner::makeDirectories(const std::string& path) {
    if (path.empty()) return;

    std::string current;
    if (path[0] == '/') current = "/";

    std::size_t start = (path[0] == '/') ? 1 : 0;
    while (start <= path.size()) {
        const std::size_t slash = path.find('/', start);
        const std::string part = path.substr(start, slash - start);
        if (!part.empty()) {
            if (!current.empty() && current[current.size() - 1] != '/') current += '/';
            current += part;
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error(
                    "mkdir(" + current + ") failed: " + std::strerror(errno));
            }
        }
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
}

std::vector<ORB_SLAM3::IMU::Point>
OrbSlamRunner::convertImu(const std::vector<LiveImuSample>& input) const {
    std::vector<ORB_SLAM3::IMU::Point> output;
    output.reserve(input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        const LiveImuSample& m = input[i];
        output.push_back(ORB_SLAM3::IMU::Point(
            m.accel_x, m.accel_y, m.accel_z,
            m.gyro_x, m.gyro_y, m.gyro_z,
            seconds(m.timestamp_ns)));
    }
    return output;
}

bool OrbSlamRunner::process(const LiveFrame& frame) {
    if (finished_ || !slam_) return false;
    if (frame.image0.empty()) {
        std::cerr << "ERROR: OrbSlamRunner received an empty primary image.\n";
        return false;
    }

    const double timestamp_s = seconds(frame.timestamp_ns);
    const double frame_dt_ms =
        (frame_count_ == 0)
            ? 0.0
            : (timestamp_s - previous_timestamp_s_) * 1000.0;

    const std::vector<ORB_SLAM3::IMU::Point> imu = convertImu(frame.imu);

    if (sensor_ == LiveSensorMode::MONO_IMU ||
        sensor_ == LiveSensorMode::STEREO_IMU) {
        const bool missing_imu = frame_count_ > 0 && frame.imu.empty();

        double first_imu_s = 0.0;
        double last_imu_s = 0.0;
        double first_after_prev_ms = 0.0;
        double cam_after_last_ms = 0.0;
        double mean_imu_dt_ms = 0.0;
        double mean_acc_norm = 0.0;
        double mean_gyro_norm = 0.0;

        if (!frame.imu.empty()) {
            first_imu_s = seconds(frame.imu.front().timestamp_ns);
            last_imu_s = seconds(frame.imu.back().timestamp_ns);

            if (frame_count_ > 0) {
                first_after_prev_ms =
                    (first_imu_s - previous_timestamp_s_) * 1000.0;
            }
            cam_after_last_ms = (timestamp_s - last_imu_s) * 1000.0;

            for (std::size_t i = 0; i < frame.imu.size(); ++i) {
                const LiveImuSample& m = frame.imu[i];
                mean_acc_norm += std::sqrt(
                    m.accel_x * m.accel_x +
                    m.accel_y * m.accel_y +
                    m.accel_z * m.accel_z);
                mean_gyro_norm += std::sqrt(
                    m.gyro_x * m.gyro_x +
                    m.gyro_y * m.gyro_y +
                    m.gyro_z * m.gyro_z);

                if (i > 0) {
                    mean_imu_dt_ms +=
                        seconds(frame.imu[i].timestamp_ns -
                                frame.imu[i - 1].timestamp_ns) * 1000.0;
                }
            }

            mean_acc_norm /= static_cast<double>(frame.imu.size());
            mean_gyro_norm /= static_cast<double>(frame.imu.size());
            if (frame.imu.size() > 1) {
                mean_imu_dt_ms /=
                    static_cast<double>(frame.imu.size() - 1);
            }
        }

        const bool timing_suspicious =
            missing_imu ||
            (!frame.imu.empty() &&
             (cam_after_last_ms < -15.0 || cam_after_last_ms > 5.0));

        if (frame_count_ < 10 ||
            ((frame_count_ + 1) % 30 == 0) ||
            timing_suspicious) {
            std::cout
                << "\n[VIO INPUT]"
                << " frame=" << frame_count_
                << " cam=" << std::fixed << std::setprecision(6)
                << timestamp_s
                << " cam_dt_ms=" << std::setprecision(3) << frame_dt_ms
                << " imu_n=" << frame.imu.size();

            if (!frame.imu.empty()) {
                std::cout
                    << " imu_first=" << std::setprecision(6) << first_imu_s
                    << " imu_last=" << last_imu_s
                    << " first_after_prev_ms=" << std::setprecision(3)
                    << first_after_prev_ms
                    << " cam_after_last_ms=" << cam_after_last_ms
                    << " imu_dt_mean_ms=" << mean_imu_dt_ms
                    << " acc_norm_mean=" << mean_acc_norm
                    << " gyro_norm_mean=" << mean_gyro_norm;
            }

            if (missing_imu) std::cout << " WARNING=no_imu_batch";
            std::cout << '\n';
        }
    }

    const std::chrono::steady_clock::time_point track_start =
        std::chrono::steady_clock::now();

    switch (sensor_) {
        case LiveSensorMode::MONO:
            slam_->TrackMonocular(frame.image0, timestamp_s);
            break;

        case LiveSensorMode::MONO_IMU:
            slam_->TrackMonocular(frame.image0, timestamp_s, imu);
            break;

        case LiveSensorMode::STEREO:
            if (frame.image1.empty()) {
                std::cerr << "ERROR: Stereo mode needs image1.\n";
                return false;
            }
            slam_->TrackStereo(frame.image0, frame.image1, timestamp_s);
            break;

        case LiveSensorMode::STEREO_IMU:
            if (frame.image1.empty()) {
                std::cerr << "ERROR: Stereo-inertial mode needs image1.\n";
                return false;
            }
            slam_->TrackStereo(frame.image0, frame.image1, timestamp_s, imu);
            break;

        case LiveSensorMode::RGBD:
            if (frame.depth.empty()) {
                std::cerr << "ERROR: RGB-D mode needs a depth image.\n";
                return false;
            }
            slam_->TrackRGBD(frame.image0, frame.depth, timestamp_s);
            break;
    }

    const std::chrono::steady_clock::time_point track_end =
        std::chrono::steady_clock::now();
    const double tracking_time_ms =
        std::chrono::duration<double, std::milli>(track_end - track_start).count();

    tracking_times_ms_.push_back(tracking_time_ms);
    updateDiagnostics(timestamp_s, frame_dt_ms, tracking_time_ms);

    previous_timestamp_s_ = timestamp_s;
    ++frame_count_;
    return true;
}

void OrbSlamRunner::updateDiagnostics(double timestamp_s,
                                      double frame_dt_ms,
                                      double tracking_time_ms) {
    const int state = slam_->GetTrackingState();
    const std::vector<cv::KeyPoint> keypoints = slam_->GetTrackedKeyPointsUn();
    const std::vector<ORB_SLAM3::MapPoint*> map_points = slam_->GetTrackedMapPoints();

    const std::size_t valid_map_points =
        static_cast<std::size_t>(std::count_if(
            map_points.begin(), map_points.end(),
            [](ORB_SLAM3::MapPoint* p) { return p != NULL; }));

    const bool slow_tracking = tracking_time_ms > SLOW_TRACKING_THRESHOLD_MS;
    if (slow_tracking) {
        ++slow_tracking_frames_;
        std::cout
            << "\n[SLOW TRACKING] frame=" << frame_count_
            << " t=" << std::fixed << std::setprecision(3) << timestamp_s << " s"
            << " state=" << trackingStateName(state)
            << " track=" << std::setprecision(1) << tracking_time_ms << " ms\n";
    }

    if (state != previous_state_) {
        std::cout
            << "\n[STATE] " << trackingStateName(previous_state_)
            << " -> " << trackingStateName(state)
            << " at t=" << std::fixed << std::setprecision(3)
            << timestamp_s << " s\n";
    }

    if (!loss_episode_active_ &&
        previous_state_ == ORB_SLAM3::Tracking::OK &&
        (state == ORB_SLAM3::Tracking::RECENTLY_LOST ||
         state == ORB_SLAM3::Tracking::LOST)) {
        loss_episode_active_ = true;
        loss_start_timestamp_s_ = timestamp_s;
        loss_start_frame_ = frame_count_;
        ++tracking_interruptions_;

        std::cout
            << "[LOSS] Tracking interruption " << tracking_interruptions_
            << " began at t=" << std::fixed << std::setprecision(3)
            << timestamp_s << " s\n";
    }

    if (loss_episode_active_ && state == ORB_SLAM3::Tracking::OK) {
        const double duration = timestamp_s - loss_start_timestamp_s_;
        const std::size_t lost_frames = frame_count_ - loss_start_frame_;
        const bool significant =
            duration >= SIGNIFICANT_LOSS_THRESHOLD_S ||
            lost_frames >= SIGNIFICANT_LOSS_MIN_FRAMES;

        if (significant) ++significant_loss_episodes_;
        else ++brief_interruptions_;
        ++recovered_episodes_;
        loss_episode_active_ = false;

        std::cout
            << "[RECOVERY] Tracking recovered at t="
            << std::fixed << std::setprecision(3) << timestamp_s << " s"
            << " | duration=" << duration << " s"
            << " | lost_frames=" << lost_frames
            << " | " << (significant ? "SIGNIFICANT" : "BRIEF") << '\n';
    }

    if (loss_episode_active_ &&
        (state == ORB_SLAM3::Tracking::NO_IMAGES_YET ||
         state == ORB_SLAM3::Tracking::NOT_INITIALIZED)) {
        const double duration = timestamp_s - loss_start_timestamp_s_;
        const std::size_t lost_frames = frame_count_ - loss_start_frame_;

        ++significant_loss_episodes_;
        ++failed_recovery_episodes_;
        loss_episode_active_ = false;

        std::cout
            << "[REINITIALISATION] Tracking recovery failed at t="
            << std::fixed << std::setprecision(3) << timestamp_s << " s"
            << " | duration=" << duration << " s"
            << " | lost_frames=" << lost_frames << '\n';
    }

    stats_file_
        << frame_count_ << ','
        << std::fixed << std::setprecision(9) << timestamp_s << ','
        << std::setprecision(3) << frame_dt_ms << ','
        << trackingStateName(state) << ','
        << tracking_time_ms << ','
        << (slow_tracking ? 1 : 0) << ','
        << keypoints.size() << ','
        << valid_map_points << '\n';

    if (slow_tracking) stats_file_.flush();

    if ((frame_count_ + 1) % 30 == 0) {
        const double mean =
            std::accumulate(tracking_times_ms_.begin(), tracking_times_ms_.end(), 0.0) /
            static_cast<double>(tracking_times_ms_.size());

        std::cout
            << "\rFrames: " << (frame_count_ + 1)
            << " | t: " << std::fixed << std::setprecision(2) << timestamp_s << " s"
            << " | state: " << trackingStateName(state)
            << " | track: " << std::setprecision(1) << tracking_time_ms << " ms"
            << " | mean: " << mean << " ms"
            << " | kp: " << keypoints.size()
            << " | MPs: " << valid_map_points
            << "       " << std::flush;
    }

    previous_state_ = state;
}

void OrbSlamRunner::finish() {
    if (finished_) return;
    finished_ = true;

    if (!slam_) return;

    std::cout << "\nStopping ORB-SLAM3...\n";
    slam_->Shutdown();

    stats_file_.flush();
    stats_file_.close();

    const std::string keyframe_path =
        joinPath(output_dir_, "KeyFrameTrajectory.txt");

    if (sensor_ == LiveSensorMode::MONO) {
        slam_->SaveKeyFrameTrajectoryTUM(keyframe_path);
    } else if (sensor_ == LiveSensorMode::MONO_IMU ||
               sensor_ == LiveSensorMode::STEREO_IMU) {
        slam_->SaveTrajectoryEuRoC(joinPath(output_dir_, "CameraTrajectory.txt"));
        slam_->SaveKeyFrameTrajectoryEuRoC(keyframe_path);
    } else {
        slam_->SaveTrajectoryTUM(joinPath(output_dir_, "CameraTrajectory.txt"));
        slam_->SaveKeyFrameTrajectoryTUM(keyframe_path);
    }

    // This call is preserved from your existing fork. It is not part of
    // upstream ORB-SLAM3; remove/comment it if building against an unmodified
    // upstream tree.
    slam_->SavePointCloudPLY(joinPath(output_dir_, "ORB_SLAM3_PointCloud"));

    printSummary();
    slam_.reset();
}

void OrbSlamRunner::printSummary() const {
    if (tracking_times_ms_.empty()) return;

    std::vector<double> sorted = tracking_times_ms_;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();

    const double mean =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) /
        static_cast<double>(n);

    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = sorted[i] - mean;
        variance += d * d;
    }
    variance /= static_cast<double>(n);

    const double median =
        (n % 2 == 0)
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            : sorted[n / 2];

    const std::size_t p95_index = std::min(
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1,
        n - 1);

    const double elapsed = previous_timestamp_s_;
    const double fps =
        (elapsed > 0.0 && frame_count_ > 1)
            ? static_cast<double>(frame_count_ - 1) / elapsed
            : 0.0;

    const double slow_percentage =
        100.0 * static_cast<double>(slow_tracking_frames_) /
        static_cast<double>(n);

    std::cout
        << "\n========== Live SLAM Statistics ==========\n"
        << "Frames processed : " << frame_count_ << '\n'
        << "Elapsed time     : " << std::fixed << std::setprecision(2)
        << elapsed << " s\n"
        << "Processed FPS    : " << fps << "\n\n"
        << "Tracking time:\n"
        << "  mean           : " << mean << " ms\n"
        << "  median         : " << median << " ms\n"
        << "  std dev        : " << std::sqrt(variance) << " ms\n"
        << "  min            : " << sorted.front() << " ms\n"
        << "  max            : " << sorted.back() << " ms\n"
        << "  95th percentile: " << sorted[p95_index] << " ms\n\n"
        << "Slow frames > " << SLOW_TRACKING_THRESHOLD_MS << " ms : "
        << slow_tracking_frames_ << " (" << slow_percentage << "%)\n\n"
        << "Tracking interruptions : " << tracking_interruptions_ << '\n'
        << "Significant losses     : " << significant_loss_episodes_ << '\n'
        << "Brief interruptions    : " << brief_interruptions_ << '\n'
        << "Recovered episodes     : " << recovered_episodes_ << '\n'
        << "Failed recoveries      : " << failed_recovery_episodes_ << '\n'
        << "Unresolved at shutdown : " << (loss_episode_active_ ? 1 : 0) << '\n'
        << "==========================================\n"
        << "Outputs: " << output_dir_ << '\n';
}
