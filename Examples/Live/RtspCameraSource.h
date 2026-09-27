#pragma once

#include "LiveSource.h"

#include <opencv2/videoio.hpp>

#include <chrono>
#include <string>

// Camera-only RTSP source migrated from the original mono_live.cc.
// It timestamps frames at receiver arrival time using steady_clock.
class RtspCameraSource : public LiveSource {
public:
    RtspCameraSource(const std::string& rtsp_url,
                     int expected_width,
                     int expected_height);
    ~RtspCameraSource();

    bool open() override;
    bool read(LiveFrame& frame) override;
    void close() override;

    bool supports(LiveSensorMode mode) const override;
    std::string description() const override;

private:
    std::string makePipeline() const;
    bool validateFrame(const cv::Mat& frame) const;

    std::string rtsp_url_;
    int expected_width_;
    int expected_height_;

    cv::VideoCapture cap_;
    cv::Mat prefetched_frame_;
    bool has_prefetched_frame_ = false;
    bool opened_ = false;

    std::chrono::steady_clock::time_point time_origin_;
};
