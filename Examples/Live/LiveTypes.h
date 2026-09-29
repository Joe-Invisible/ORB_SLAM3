#pragma once

#include <opencv2/core/core.hpp>

#include <cstdint>
#include <vector>

// Sensor mode describes what a LiveFrame contains and how ORB-SLAM3 should
// consume it. Acquisition backends advertise the modes they actually provide,
// while the downstream interface remains shared across camera/IMU combinations.
enum class LiveSensorMode {
    MONO,
    MONO_IMU,
    STEREO,
    STEREO_IMU,
    RGBD
};

inline const char* LiveSensorModeName(LiveSensorMode mode) {
    switch (mode) {
        case LiveSensorMode::MONO:       return "mono";
        case LiveSensorMode::MONO_IMU:   return "mono-imu";
        case LiveSensorMode::STEREO:     return "stereo";
        case LiveSensorMode::STEREO_IMU: return "stereo-imu";
        case LiveSensorMode::RGBD:       return "rgbd";
        default:                         return "unknown";
    }
}

struct LiveImuSample {
    std::int64_t timestamp_ns = 0;

    double gyro_x = 0.0;
    double gyro_y = 0.0;
    double gyro_z = 0.0;

    double accel_x = 0.0;
    double accel_y = 0.0;
    double accel_z = 0.0;
};

// Optional acquisition diagnostics. These timestamps are intentionally kept
// separate from LiveFrame::timestamp_ns because they may live in different
// clock domains. They are for timing analysis, not direct SLAM consumption.
struct LiveCameraTiming {
    bool valid = false;

    // Raw GStreamer timestamps carried by the decoded video buffer.
    std::int64_t source_pts_ns = -1;
    std::int64_t source_dts_ns = -1;

    // Host steady_clock timestamps around appsink delivery / image copy.
    std::int64_t host_pull_ns = -1;
    std::int64_t host_copy_done_ns = -1;

    // Same host_pull event mapped into the SensorLog phone clock.
    std::int64_t host_mapped_phone_ns = -1;

    // Camera timestamp in the SensorLog phone clock before session-origin
    // subtraction. This includes the configured constant camera offset.
    std::int64_t frame_phone_time_ns = -1;
};

struct LiveFrame {
    // All sensor streams in one LiveFrame must use the same clock domain.
    // timestamp_ns is relative to the source's session origin.
    std::int64_t timestamp_ns = 0;

    // MONO / MONO_IMU: image0 only
    // STEREO / STEREO_IMU: image0=left, image1=right
    // RGBD: image0=RGB/gray, depth=registered depth image
    cv::Mat image0;
    cv::Mat image1;
    cv::Mat depth;

    // IMU samples since the previous camera frame, up to this frame timestamp.
    std::vector<LiveImuSample> imu;

    // Optional source-level timing diagnostics.
    LiveCameraTiming camera_timing;
};
