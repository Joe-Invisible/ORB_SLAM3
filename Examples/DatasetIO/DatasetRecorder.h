#pragma once

#include <opencv2/core/core.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

// SLAM-independent recorder for a timestamped monocular camera stream and
// an optional IMU stream. The recorder has no dependency on ORB-SLAM3.
//
// All timestamps must use one shared clock domain and are stored exactly as
// signed integer nanoseconds. The caller decides which streams exist:
//   visual-only:       call recordFrame()
//   visual-inertial:   call recordFrame() and recordImu()
class DatasetRecorder {
public:
    struct Options {
        Options() : max_pending_frames(120), png_compression(1) {}
        std::size_t max_pending_frames;
        int png_compression; // OpenCV PNG compression: 0 (fast) .. 9 (small)
    };

    explicit DatasetRecorder(const std::string& dataset_dir,
                             const Options& options = Options());
    ~DatasetRecorder();

    DatasetRecorder(const DatasetRecorder&) = delete;
    DatasetRecorder& operator=(const DatasetRecorder&) = delete;

    // Records the exact image handed to the downstream SLAM frontend.
    // The image is cloned before this function returns; PNG encoding happens
    // on a worker thread. If the queue fills, the producer blocks rather than
    // silently dropping frames.
    bool recordFrame(std::int64_t timestamp_ns, const cv::Mat& frame);

    // Optional. imu.csv is created lazily on the first IMU sample, so a
    // visual-only recording contains no misleading empty IMU stream.
    // Units: gyroscope rad/s; acceleration m/s^2.
    bool recordImu(std::int64_t timestamp_ns,
                   double gyro_x, double gyro_y, double gyro_z,
                   double accel_x, double accel_y, double accel_z);

    // Flushes queued frames, closes files, and writes session_info.txt.
    // Safe to call more than once.
    bool finish();

    bool good() const;
    std::string errorMessage() const;
    std::size_t frameCount() const;
    std::size_t imuCount() const;

private:
    struct PendingFrame {
        std::int64_t timestamp_ns;
        std::string relative_path;
        cv::Mat image;
    };

    void writerLoop();
    bool ensureImuFileOpen();
    void writeSessionInfo();
    void setError(const std::string& message);
    static bool makeDirectories(const std::string& path);

    const std::string dataset_dir_;
    const std::string camera_dir_;
    const Options options_;

    std::ofstream camera_csv_;
    std::ofstream imu_csv_;

    mutable std::mutex state_mutex_;
    std::mutex imu_mutex_;
    std::condition_variable queue_not_empty_;
    std::condition_variable queue_not_full_;
    std::deque<PendingFrame> frame_queue_;
    std::thread writer_thread_;

    bool stopping_ = false;
    bool finished_ = false;
    bool ok_ = true;
    bool imu_file_opened_ = false;
    std::string error_message_;

    std::int64_t last_camera_timestamp_ns_ = -1;
    std::int64_t last_imu_timestamp_ns_ = -1;
    std::size_t frame_count_ = 0;
    std::size_t imu_count_ = 0;
};
