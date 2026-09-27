#pragma once

#include "LiveTypes.h"

#include <System.h>
#include <Tracking.h>

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// ORB-SLAM3 consumer for LiveFrame. Acquisition and recording know nothing
// about this class, so the same live source can be record-only or SLAM+record.
class OrbSlamRunner {
public:
    OrbSlamRunner(LiveSensorMode sensor,
                  const std::string& vocabulary_path,
                  const std::string& settings_path,
                  bool use_viewer,
                  const std::string& output_dir);
    ~OrbSlamRunner();

    OrbSlamRunner(const OrbSlamRunner&) = delete;
    OrbSlamRunner& operator=(const OrbSlamRunner&) = delete;

    bool process(const LiveFrame& frame);
    void finish();

    std::size_t frameCount() const { return frame_count_; }

private:
    static ORB_SLAM3::System::eSensor toOrbSensor(LiveSensorMode sensor);
    static const char* trackingStateName(int state);
    static double seconds(std::int64_t timestamp_ns);
    static std::string joinPath(const std::string& a, const std::string& b);
    static void makeDirectories(const std::string& path);

    std::vector<ORB_SLAM3::IMU::Point>
    convertImu(const std::vector<LiveImuSample>& input) const;

    void updateDiagnostics(double timestamp_s,
                           double frame_dt_ms,
                           double tracking_time_ms);
    void printSummary() const;

    LiveSensorMode sensor_;
    std::string output_dir_;
    std::unique_ptr<ORB_SLAM3::System> slam_;
    std::ofstream stats_file_;

    bool finished_ = false;
    std::size_t frame_count_ = 0;
    double previous_timestamp_s_ = 0.0;

    std::vector<double> tracking_times_ms_;
    std::size_t slow_tracking_frames_ = 0;

    std::size_t tracking_interruptions_ = 0;
    std::size_t significant_loss_episodes_ = 0;
    std::size_t brief_interruptions_ = 0;
    std::size_t recovered_episodes_ = 0;
    std::size_t failed_recovery_episodes_ = 0;

    bool loss_episode_active_ = false;
    double loss_start_timestamp_s_ = 0.0;
    std::size_t loss_start_frame_ = 0;
    int previous_state_ = ORB_SLAM3::Tracking::NO_IMAGES_YET;
};
