#include "RtspMonoImuSource.h"

#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
constexpr GstClockTime kPullTimeout = 2 * GST_SECOND;
}

RtspMonoImuSource::RtspMonoImuSource(const std::string& rtsp_url,
                                     int expected_width,
                                     int expected_height,
                                     int imu_port,
                                     int imu_timeout_ms,
                                     double camera_time_offset_ms)
    : rtsp_url_(rtsp_url),
      expected_width_(expected_width),
      expected_height_(expected_height),
      imu_timeout_ms_(imu_timeout_ms),
      camera_time_offset_s_(camera_time_offset_ms * 1e-3),
      imu_receiver_(imu_port) {}

RtspMonoImuSource::~RtspMonoImuSource() { close(); }

std::int64_t RtspMonoImuSource::hostSteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string RtspMonoImuSource::makePipeline() const {
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
        "appsink name=framesink "
        "max-buffers=1 "
        "drop=true "
        "sync=false "
        "emit-signals=false";
}

bool RtspMonoImuSource::validateFrame(const cv::Mat& frame) const {
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

bool RtspMonoImuSource::hostTimeAsPhoneTime(std::int64_t host_time_ns,
                                            double& phone_time_s) const {
    return imu_receiver_.phoneTimeForHostNs(host_time_ns, phone_time_s);
}

bool RtspMonoImuSource::ptsAsPhoneTime(std::int64_t pts_ns,
                                       double& frame_phone_time_s) const {
    if (!have_pts_phone_anchor_ || pts_ns < 0) return false;

    frame_phone_time_s =
        static_cast<double>(pts_ns) * 1e-9 +
        pts_to_phone_offset_s_ -
        camera_time_offset_s_;
    return true;
}

void RtspMonoImuSource::reportPipelineError(const char* prefix) const {
    if (!pipeline_) return;

    GstBus* bus = gst_element_get_bus(pipeline_);
    if (!bus) return;

    GstMessage* message = gst_bus_pop_filtered(
        bus,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (message) {
        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
            GError* error = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(message, &error, &debug);

            std::cerr << prefix;
            if (error && error->message) std::cerr << ": " << error->message;
            std::cerr << '\n';

            if (debug) {
                std::cerr << "GStreamer debug: " << debug << '\n';
            }

            if (error) g_error_free(error);
            if (debug) g_free(debug);
        } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
            std::cerr << prefix << ": end of RTSP stream.\n";
        }
        gst_message_unref(message);
    }

    gst_object_unref(bus);
}

bool RtspMonoImuSource::openPipeline() {
    gst_init(nullptr, nullptr);

    GError* error = nullptr;
    pipeline_ = gst_parse_launch(makePipeline().c_str(), &error);
    if (!pipeline_) {
        std::cerr << "ERROR: Could not construct GStreamer RTSP pipeline";
        if (error && error->message) std::cerr << ": " << error->message;
        std::cerr << '\n';
        if (error) g_error_free(error);
        return false;
    }

    if (error) {
        std::cerr << "WARNING: GStreamer pipeline parser: "
                  << (error->message ? error->message : "unknown warning")
                  << '\n';
        g_error_free(error);
        error = nullptr;
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "framesink");
    if (!appsink_) {
        std::cerr << "ERROR: GStreamer pipeline does not contain framesink.\n";
        return false;
    }

    const GstStateChangeReturn state_result =
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (state_result == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "ERROR: Could not start GStreamer RTSP pipeline.\n";
        reportPipelineError("GStreamer");
        return false;
    }

    return true;
}

bool RtspMonoImuSource::pullFrame(PulledFrame& out) {
    if (!appsink_) return false;

    GstSample* sample =
        gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), kPullTimeout);

    // This is deliberately taken immediately after appsink returns a decoded
    // sample. It is the closest analogue of the old "time after cap.read()"
    // diagnostic, but it is no longer used to timestamp every camera frame.
    const std::int64_t host_pull_ns = hostSteadyNowNs();

    if (!sample) {
        if (gst_app_sink_is_eos(GST_APP_SINK(appsink_))) {
            std::cerr << "WARNING: RTSP appsink reached EOS.\n";
        } else {
            std::cerr << "WARNING: Timed out waiting for RTSP appsink sample.\n";
        }
        reportPipelineError("GStreamer");
        return false;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    if (!buffer || !caps) {
        std::cerr << "ERROR: GStreamer sample is missing buffer/caps.\n";
        gst_sample_unref(sample);
        return false;
    }

    const GstClockTime pts = GST_BUFFER_PTS(buffer);
    const GstClockTime dts = GST_BUFFER_DTS(buffer);

    if (!GST_CLOCK_TIME_IS_VALID(pts) ||
        pts > static_cast<GstClockTime>(std::numeric_limits<std::int64_t>::max())) {
        std::cerr << "ERROR: RTSP frame has no usable GStreamer PTS.\n";
        gst_sample_unref(sample);
        return false;
    }

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        std::cerr << "ERROR: Could not parse GStreamer video caps.\n";
        gst_sample_unref(sample);
        return false;
    }

    const int width = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
    const int height = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
    const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
    const gsize offset = GST_VIDEO_INFO_PLANE_OFFSET(&info, 0);

    if (width <= 0 || height <= 0 || stride <= 0) {
        std::cerr << "ERROR: Invalid GStreamer BGR frame geometry.\n";
        gst_sample_unref(sample);
        return false;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        std::cerr << "ERROR: Could not map GStreamer video buffer.\n";
        gst_sample_unref(sample);
        return false;
    }

    const std::size_t required =
        static_cast<std::size_t>(offset) +
        static_cast<std::size_t>(stride) *
        static_cast<std::size_t>(height);

    if (map.size < required) {
        std::cerr << "ERROR: GStreamer video buffer is smaller than its caps imply.\n";
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return false;
    }

    const unsigned char* pixels = map.data + offset;
    cv::Mat wrapped(height, width, CV_8UC3,
                    const_cast<unsigned char*>(pixels),
                    static_cast<std::size_t>(stride));

    out = PulledFrame();
    out.image = wrapped.clone();
    out.host_pull_ns = host_pull_ns;
    out.host_copy_done_ns = hostSteadyNowNs();
    out.pts_ns = static_cast<std::int64_t>(pts);

    if (GST_CLOCK_TIME_IS_VALID(dts) &&
        dts <= static_cast<GstClockTime>(std::numeric_limits<std::int64_t>::max())) {
        out.dts_ns = static_cast<std::int64_t>(dts);
    }

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return validateFrame(out.image);
}

void RtspMonoImuSource::populateTimingDiagnostics(
    const PulledFrame& pulled,
    double frame_phone_time_s,
    LiveFrame& frame) const {

    double host_mapped_phone_s = 0.0;
    if (!hostTimeAsPhoneTime(pulled.host_pull_ns, host_mapped_phone_s)) {
        return;
    }

    frame.camera_timing.valid = true;
    frame.camera_timing.source_pts_ns = pulled.pts_ns;
    frame.camera_timing.source_dts_ns = pulled.dts_ns;
    frame.camera_timing.host_pull_ns = pulled.host_pull_ns;
    frame.camera_timing.host_copy_done_ns = pulled.host_copy_done_ns;
    frame.camera_timing.host_mapped_phone_ns = static_cast<std::int64_t>(
        std::llround(host_mapped_phone_s * 1e9));
    frame.camera_timing.frame_phone_time_ns = static_cast<std::int64_t>(
        std::llround(frame_phone_time_s * 1e9));
}

bool RtspMonoImuSource::open() {
    if (opened_) return true;

    std::cout << "Starting SensorLog IMU receiver...\n";
    if (!imu_receiver_.start()) {
        std::cerr << "ERROR: " << imu_receiver_.errorMessage() << '\n';
        return false;
    }

    std::cout << "Waiting for SensorLog IMU clock warm-up...\n";
    if (!imu_receiver_.waitUntilReady(5000)) {
        std::cerr
            << "ERROR: Timed out waiting for synchronized SensorLog IMU data.\n"
            << "Check that SensorLog is streaming UDP to this machine.\n";
        close();
        return false;
    }

    std::cout << "Opening RTSP stream through native GStreamer appsink...\n";
    if (!openPipeline()) {
        close();
        return false;
    }

    std::cout << "RTSP stream opened. Waiting for first PTS-stamped frame...\n";
    if (!pullFrame(prefetched_frame_)) {
        close();
        return false;
    }

    double first_arrival_phone_time_s = 0.0;
    if (!hostTimeAsPhoneTime(
            prefetched_frame_.host_pull_ns,
            first_arrival_phone_time_s)) {
        std::cerr << "ERROR: Could not map first camera arrival into SensorLog clock.\n";
        close();
        return false;
    }

    // Anchor the arbitrary GStreamer PTS origin once. This intentionally uses
    // only the first frame's arrival time. Any error here becomes a single
    // constant camera/IMU offset; subsequent per-frame network/decode jitter
    // does not enter the camera timestamps.
    pts_to_phone_offset_s_ =
        first_arrival_phone_time_s -
        static_cast<double>(prefetched_frame_.pts_ns) * 1e-9;
    have_pts_phone_anchor_ = true;

    if (!ptsAsPhoneTime(prefetched_frame_.pts_ns, phone_time_origin_s_)) {
        std::cerr << "ERROR: Could not map first camera PTS into SensorLog clock.\n";
        close();
        return false;
    }

    previous_frame_phone_time_s_ = phone_time_origin_s_;
    previous_pts_ns_ = prefetched_frame_.pts_ns;
    imu_receiver_.discardThrough(phone_time_origin_s_);

    has_prefetched_frame_ = true;
    opened_ = true;

    std::cout
        << "First PTS-stamped camera frame received: "
        << prefetched_frame_.image.cols << 'x'
        << prefetched_frame_.image.rows << '\n'
        << "First GstBuffer PTS: "
        << std::fixed << std::setprecision(3)
        << static_cast<double>(prefetched_frame_.pts_ns) * 1e-6 << " ms\n"
        << "Camera time offset: "
        << camera_time_offset_s_ * 1000.0 << " ms\n"
        << "Per-frame camera timing now follows GstBuffer PTS, not host arrival.\n";
    return true;
}

bool RtspMonoImuSource::read(LiveFrame& frame) {
    if (!opened_) return false;

    PulledFrame pulled;

    if (has_prefetched_frame_) {
        pulled = prefetched_frame_;
        prefetched_frame_ = PulledFrame();
        has_prefetched_frame_ = false;

        frame = LiveFrame();
        frame.timestamp_ns = 0;
        frame.image0 = pulled.image;
        populateTimingDiagnostics(
            pulled,
            phone_time_origin_s_,
            frame);

        // The first image establishes t=0. IMU batching begins after it.
        return true;
    }

    if (!pullFrame(pulled)) return false;

    if (pulled.pts_ns <= previous_pts_ns_) {
        std::cerr
            << "ERROR: Non-monotonic GStreamer camera PTS: "
            << previous_pts_ns_ << " -> " << pulled.pts_ns << '\n';
        return false;
    }

    double frame_phone_time_s = 0.0;
    if (!ptsAsPhoneTime(pulled.pts_ns, frame_phone_time_s)) {
        std::cerr << "ERROR: Lost the GStreamer PTS/phone clock mapping.\n";
        return false;
    }

    if (frame_phone_time_s <= previous_frame_phone_time_s_) {
        std::cerr << "ERROR: Non-monotonic synchronized camera timestamp.\n";
        return false;
    }

    if (!imu_receiver_.waitUntilPhoneTime(frame_phone_time_s, imu_timeout_ms_)) {
        std::cerr
            << "ERROR: IMU stream did not reach camera timestamp within "
            << imu_timeout_ms_ << " ms.\n";
        const std::string receiver_error = imu_receiver_.errorMessage();
        if (!receiver_error.empty()) std::cerr << receiver_error << '\n';
        return false;
    }

    frame = LiveFrame();
    frame.timestamp_ns = static_cast<std::int64_t>(
        std::llround((frame_phone_time_s - phone_time_origin_s_) * 1e9));
    frame.image0 = pulled.image;
    imu_receiver_.popThrough(
        frame_phone_time_s,
        phone_time_origin_s_,
        frame.imu);
    populateTimingDiagnostics(pulled, frame_phone_time_s, frame);

    previous_pts_ns_ = pulled.pts_ns;
    previous_frame_phone_time_s_ = frame_phone_time_s;
    return true;
}

void RtspMonoImuSource::close() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    }

    if (appsink_) {
        gst_object_unref(appsink_);
        appsink_ = nullptr;
    }

    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }

    prefetched_frame_ = PulledFrame();
    has_prefetched_frame_ = false;
    have_pts_phone_anchor_ = false;
    previous_pts_ns_ = -1;
    opened_ = false;
    imu_receiver_.stop();
}

bool RtspMonoImuSource::supports(LiveSensorMode mode) const {
    return mode == LiveSensorMode::MONO_IMU;
}

std::string RtspMonoImuSource::description() const {
    std::ostringstream os;
    os << "RTSP/GStreamer-PTS camera + SensorLog UDP IMU: " << rtsp_url_;
    return os.str();
}
