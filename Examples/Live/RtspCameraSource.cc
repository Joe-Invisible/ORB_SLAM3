#include "RtspCameraSource.h"

#include <iostream>
#include <sstream>

RtspCameraSource::RtspCameraSource(const std::string& rtsp_url,
                                   int expected_width,
                                   int expected_height)
    : rtsp_url_(rtsp_url),
      expected_width_(expected_width),
      expected_height_(expected_height) {}

RtspCameraSource::~RtspCameraSource() {
    close();
}

std::string RtspCameraSource::makePipeline() const {
    // Preserve the low-latency pipeline from the original mono_live.cc.
    return
        "rtspsrc location=\"" + rtsp_url_ + "\" "
        "protocols=udp "
        "latency=50 "
        "drop-on-latency=true ! "
        "rtph264depay ! "
        "h264parse ! "
        "avdec_h264 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false";
}

bool RtspCameraSource::validateFrame(const cv::Mat& frame) const {
    if (frame.empty()) {
        std::cerr << "ERROR: Received empty camera frame.\n";
        return false;
    }

    if (frame.cols != expected_width_ || frame.rows != expected_height_) {
        std::cerr
            << "ERROR: Unexpected stream resolution: "
            << frame.cols << 'x' << frame.rows << '\n'
            << "Expected " << expected_width_ << 'x' << expected_height_ << ".\n"
            << "The selected camera calibration would therefore be invalid.\n";
        return false;
    }

    return true;
}

bool RtspCameraSource::open() {
    if (opened_) return true;

    std::cout << "Opening RTSP stream...\n";
    cap_.open(makePipeline(), cv::CAP_GSTREAMER);

    if (!cap_.isOpened()) {
        std::cerr << "ERROR: Could not open RTSP stream.\n";
        return false;
    }

    std::cout << "RTSP stream opened.\nWaiting for first frame...\n";

    if (!cap_.read(prefetched_frame_) || !validateFrame(prefetched_frame_)) {
        cap_.release();
        return false;
    }

    // Define t=0 at the first frame accepted by this frontend. Recording and
    // SLAM therefore receive exactly the same time base.
    time_origin_ = std::chrono::steady_clock::now();
    has_prefetched_frame_ = true;
    opened_ = true;

    std::cout << "First frame received: "
              << prefetched_frame_.cols << 'x' << prefetched_frame_.rows << '\n';
    return true;
}

bool RtspCameraSource::read(LiveFrame& frame) {
    if (!opened_) return false;

    cv::Mat image;
    std::chrono::steady_clock::time_point frame_time;

    if (has_prefetched_frame_) {
        image = prefetched_frame_;
        prefetched_frame_.release();
        has_prefetched_frame_ = false;
        frame_time = time_origin_;
    } else {
        if (!cap_.read(image)) {
            std::cerr << "WARNING: Failed to retrieve RTSP frame.\n";
            return false;
        }
        if (image.empty()) {
            std::cerr << "WARNING: Received empty RTSP frame.\n";
            return false;
        }
        frame_time = std::chrono::steady_clock::now();
    }

    const std::int64_t timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            frame_time - time_origin_).count();

    frame = LiveFrame();
    frame.timestamp_ns = timestamp_ns;
    frame.image0 = image;
    return true;
}

void RtspCameraSource::close() {
    if (cap_.isOpened()) cap_.release();
    prefetched_frame_.release();
    has_prefetched_frame_ = false;
    opened_ = false;
}

bool RtspCameraSource::supports(LiveSensorMode mode) const {
    return mode == LiveSensorMode::MONO;
}

std::string RtspCameraSource::description() const {
    std::ostringstream os;
    os << "RTSP camera: " << rtsp_url_;
    return os.str();
}
