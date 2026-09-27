#pragma once

#include <opencv2/core/core.hpp>

#include <cstdint>
#include <vector>

// Sensor mode describes what a LiveFrame contains and how ORB-SLAM3 should
// consume it. The current RTSP source only produces MONO. The remaining modes
// are deliberately represented here so future sources can plug into the same
// main loop without redesigning the downstream interface.
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

struct LiveFrame {
    // All streams in one LiveFrame must use the same clock domain.
    // The current RTSP source uses nanoseconds since the first received frame.
    std::int64_t timestamp_ns = 0;

    // MONO / MONO_IMU: image0 only
    // STEREO / STEREO_IMU: image0=left, image1=right
    // RGBD: image0=RGB/gray, depth=registered depth image
    cv::Mat image0;
    cv::Mat image1;
    cv::Mat depth;

    // IMU samples since the previous camera frame, up to this frame timestamp.
    std::vector<LiveImuSample> imu;
};
