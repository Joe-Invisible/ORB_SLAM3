#include "DatasetRecorder.h"

#include <opencv2/imgcodecs.hpp>

#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {
std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a[a.size() - 1] == '/') return a + b;
    return a + "/" + b;
}
}

DatasetRecorder::DatasetRecorder(const std::string& dataset_dir,
                                 const Options& options)
    : dataset_dir_(dataset_dir),
      camera_dir_(joinPath(dataset_dir, "camera")),
      options_(options) {
    if (!makeDirectories(camera_dir_)) {
        setError("Failed to create dataset directory: " + camera_dir_);
        return;
    }

    camera_csv_.open(joinPath(dataset_dir_, "camera.csv").c_str(),
                     std::ios::out | std::ios::trunc);
    if (!camera_csv_) {
        setError("Failed to open camera.csv in " + dataset_dir_);
        return;
    }
    camera_csv_ << "timestamp_ns,filename\n";

    writer_thread_ = std::thread(&DatasetRecorder::writerLoop, this);
}

DatasetRecorder::~DatasetRecorder() {
    finish();
}

bool DatasetRecorder::recordFrame(std::int64_t timestamp_ns, const cv::Mat& frame) {
    if (frame.empty()) {
        setError("recordFrame() received an empty image");
        return false;
    }

    std::unique_lock<std::mutex> lock(state_mutex_);
    if (!ok_ || stopping_ || finished_) return false;

    if (last_camera_timestamp_ns_ >= 0 && timestamp_ns <= last_camera_timestamp_ns_) {
        lock.unlock();
        setError("Camera timestamps are not strictly increasing");
        return false;
    }

    queue_not_full_.wait(lock, [this] {
        return !ok_ || stopping_ || frame_queue_.size() < options_.max_pending_frames;
    });
    if (!ok_ || stopping_) return false;

    std::ostringstream name;
    name << timestamp_ns << ".png";

    PendingFrame item;
    item.timestamp_ns = timestamp_ns;
    item.relative_path = std::string("camera/") + name.str();
    item.image = frame.clone();

    frame_queue_.push_back(std::move(item));
    last_camera_timestamp_ns_ = timestamp_ns;
    lock.unlock();
    queue_not_empty_.notify_one();
    return true;
}

bool DatasetRecorder::ensureImuFileOpen() {
    if (imu_file_opened_) return true;

    imu_csv_.open(joinPath(dataset_dir_, "imu.csv").c_str(),
                  std::ios::out | std::ios::trunc);
    if (!imu_csv_) {
        return false;
    }
    imu_csv_ << "timestamp_ns,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n";
    imu_csv_ << std::setprecision(17);
    imu_file_opened_ = true;
    return true;
}

bool DatasetRecorder::recordImu(std::int64_t timestamp_ns,
                                double gyro_x, double gyro_y, double gyro_z,
                                double accel_x, double accel_y, double accel_z) {
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!ok_ || stopping_ || finished_) return false;
    }

    std::lock_guard<std::mutex> imu_lock(imu_mutex_);

    if (last_imu_timestamp_ns_ >= 0 && timestamp_ns <= last_imu_timestamp_ns_) {
        setError("IMU timestamps are not strictly increasing");
        return false;
    }

    if (!ensureImuFileOpen()) {
        setError("Failed to open imu.csv in " + dataset_dir_);
        return false;
    }

    imu_csv_ << timestamp_ns << ','
             << gyro_x << ',' << gyro_y << ',' << gyro_z << ','
             << accel_x << ',' << accel_y << ',' << accel_z << '\n';
    if (!imu_csv_) {
        setError("Failed while writing imu.csv");
        return false;
    }

    last_imu_timestamp_ns_ = timestamp_ns;
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        ++imu_count_;
    }
    return true;
}

bool DatasetRecorder::finish() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (finished_) return ok_;
        stopping_ = true;
    }
    queue_not_empty_.notify_all();
    queue_not_full_.notify_all();

    if (writer_thread_.joinable()) writer_thread_.join();

    {
        std::lock_guard<std::mutex> imu_lock(imu_mutex_);
        if (imu_file_opened_) {
            imu_csv_.flush();
            imu_csv_.close();
        }
    }

    camera_csv_.flush();
    camera_csv_.close();
    writeSessionInfo();

    std::lock_guard<std::mutex> lock(state_mutex_);
    finished_ = true;
    return ok_;
}

bool DatasetRecorder::good() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ok_;
}

std::string DatasetRecorder::errorMessage() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return error_message_;
}

std::size_t DatasetRecorder::frameCount() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return frame_count_;
}

std::size_t DatasetRecorder::imuCount() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return imu_count_;
}

void DatasetRecorder::writerLoop() {
    const std::vector<int> png_params = {
        cv::IMWRITE_PNG_COMPRESSION, options_.png_compression
    };

    for (;;) {
        PendingFrame item;
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            queue_not_empty_.wait(lock, [this] {
                return !ok_ || stopping_ || !frame_queue_.empty();
            });

            if ((!ok_ || stopping_) && frame_queue_.empty()) break;

            item = std::move(frame_queue_.front());
            frame_queue_.pop_front();
        }
        queue_not_full_.notify_one();

        const std::string absolute_path = joinPath(dataset_dir_, item.relative_path);
        if (!cv::imwrite(absolute_path, item.image, png_params)) {
            setError("cv::imwrite failed for " + absolute_path);
            break;
        }

        camera_csv_ << item.timestamp_ns << ',' << item.relative_path << '\n';
        if (!camera_csv_) {
            setError("Failed while writing camera.csv");
            break;
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        ++frame_count_;
    }
}

void DatasetRecorder::writeSessionInfo() {
    std::size_t frames = 0;
    std::size_t imus = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        frames = frame_count_;
        imus = imu_count_;
    }

    std::ofstream info(joinPath(dataset_dir_, "session_info.txt").c_str(),
                       std::ios::out | std::ios::trunc);
    if (!info) return;

    info << "format_version=2\n"
         << "timestamp_unit=ns\n"
         << "timestamp_clock=shared_monotonic_or_synchronized_sensor_clock\n"
         << "camera_encoding=lossless_png\n"
         << "camera_frames=" << frames << "\n"
         << "has_imu=" << (imus > 0 ? 1 : 0) << "\n"
         << "imu_samples=" << imus << "\n";

    if (imus > 0) {
        info << "imu_gyro_unit=rad/s\n"
             << "imu_accel_unit=m/s^2\n"
             << "imu_column_order=timestamp,gyro,accel\n";
    }
}

void DatasetRecorder::setError(const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (ok_) {
            ok_ = false;
            error_message_ = message;
            std::cerr << "[DatasetRecorder] ERROR: " << message << std::endl;
        }
    }
    queue_not_empty_.notify_all();
    queue_not_full_.notify_all();
}

bool DatasetRecorder::makeDirectories(const std::string& path) {
    if (path.empty()) return false;

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
                std::cerr << "mkdir(" << current << ") failed: "
                          << std::strerror(errno) << std::endl;
                return false;
            }
        }
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    return true;
}
