#pragma once

#include "LiveSource.h"
#include "SensorLogImuReceiver.h"

#include <gst/gst.h>

#include <opencv2/core/core.hpp>

#include <cstdint>
#include <string>

// Combined source for the current phone setup:
//   camera: RTSP/H.264 pulled directly from a GStreamer appsink
//   IMU:    SensorLog JSON over UDP
//
// Unlike the earlier OpenCV VideoCapture path, camera cadence is taken from
// GstBuffer PTS. PTS is anchored once to the SensorLog/phone clock using the
// first decoded frame's host arrival time. After that, frame-to-frame timing is
// driven only by PTS, so network/decode arrival jitter does not perturb the
// camera timestamps.
//
// camera_time_offset_ms remains a constant calibration term:
//   positive => physical exposure occurred before the PTS anchor would place it.
class RtspMonoImuSource : public LiveSource {
public:
    RtspMonoImuSource(const std::string& rtsp_url,
                      int expected_width,
                      int expected_height,
                      int imu_port,
                      int imu_timeout_ms,
                      double camera_time_offset_ms);
    ~RtspMonoImuSource();

    bool open() override;
    bool read(LiveFrame& frame) override;
    void close() override;

    bool supports(LiveSensorMode mode) const override;
    std::string description() const override;

private:
    struct PulledFrame {
        cv::Mat image;
        std::int64_t pts_ns = -1;
        std::int64_t dts_ns = -1;
        std::int64_t host_pull_ns = -1;
        std::int64_t host_copy_done_ns = -1;
    };

    std::string makePipeline() const;
    bool openPipeline();
    bool pullFrame(PulledFrame& out);
    bool validateFrame(const cv::Mat& frame) const;
    bool hostTimeAsPhoneTime(std::int64_t host_time_ns,
                             double& phone_time_s) const;
    bool ptsAsPhoneTime(std::int64_t pts_ns,
                        double& frame_phone_time_s) const;
    void populateTimingDiagnostics(const PulledFrame& pulled,
                                   double frame_phone_time_s,
                                   LiveFrame& frame) const;
    void reportPipelineError(const char* prefix) const;
    static std::int64_t hostSteadyNowNs();

    std::string rtsp_url_;
    int expected_width_;
    int expected_height_;
    int imu_timeout_ms_;
    double camera_time_offset_s_;

    SensorLogImuReceiver imu_receiver_;

    GstElement* pipeline_ = nullptr;
    GstElement* appsink_ = nullptr;

    PulledFrame prefetched_frame_;
    bool has_prefetched_frame_ = false;
    bool opened_ = false;

    // phone_time ~= PTS + pts_to_phone_offset_s_ - camera_time_offset_s_
    double pts_to_phone_offset_s_ = 0.0;
    bool have_pts_phone_anchor_ = false;

    double phone_time_origin_s_ = 0.0;
    double previous_frame_phone_time_s_ = 0.0;
    std::int64_t previous_pts_ns_ = -1;
};
