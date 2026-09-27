#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CameraSample {
    std::int64_t timestamp_ns;
    std::string filename;
};

struct ImuSample {
    std::int64_t timestamp_ns;
    double gyro_x, gyro_y, gyro_z;
    double accel_x, accel_y, accel_z;
};

// Generic reader for the native dataset format. It knows nothing about
// ORB-SLAM3; adapters can consume the same data for other algorithms later.
class DatasetReader {
public:
    explicit DatasetReader(const std::string& dataset_dir);

    const std::string& datasetDir() const { return dataset_dir_; }
    const std::vector<CameraSample>& camera() const { return camera_; }
    const std::vector<ImuSample>& imu() const { return imu_; }
    bool hasImu() const { return !imu_.empty(); }

    std::string absolutePath(const std::string& relative_path) const;

private:
    static std::vector<CameraSample> loadCameraCsv(const std::string& path);
    static std::vector<ImuSample> loadImuCsvIfPresent(const std::string& path);
    static void validateCamera(const std::vector<CameraSample>& camera);
    static void validateImu(const std::vector<ImuSample>& imu);

    std::string dataset_dir_;
    std::vector<CameraSample> camera_;
    std::vector<ImuSample> imu_;
};
