#include "DatasetReader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a[a.size() - 1] == '/') return a + b;
    return a + "/" + b;
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) fields.push_back(item);
    return fields;
}

bool isCommentOrBlank(const std::string& line) {
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == ' ' || line[i] == '\t' || line[i] == '\r') continue;
        return line[i] == '#';
    }
    return true;
}
}

DatasetReader::DatasetReader(const std::string& dataset_dir)
    : dataset_dir_(dataset_dir),
      camera_(loadCameraCsv(joinPath(dataset_dir, "camera.csv"))),
      imu_(loadImuCsvIfPresent(joinPath(dataset_dir, "imu.csv"))) {
    raw_gyro_ = loadRawCsv(joinPath(dataset_dir, "gyro.csv"));
    raw_accel_ = loadRawCsv(joinPath(dataset_dir, "accel.csv"));
    validateCamera(camera_);
    validateImu(imu_);
}

std::string DatasetReader::absolutePath(const std::string& relative_path) const {
    return joinPath(dataset_dir_, relative_path);
}

std::vector<CameraSample> DatasetReader::loadCameraCsv(const std::string& path) {
    std::ifstream f(path.c_str());
    if (!f) throw std::runtime_error("Cannot open " + path);

    std::vector<CameraSample> out;
    std::string line;
    bool first_data_line = true;

    while (std::getline(f, line)) {
        if (isCommentOrBlank(line)) continue;
        const std::vector<std::string> c = splitCsv(line);
        if (first_data_line) {
            first_data_line = false;
            if (!c.empty() && c[0].find("timestamp") != std::string::npos) continue;
        }
        if (c.size() < 2) throw std::runtime_error("Bad camera.csv row: " + line);

        CameraSample sample;
        sample.timestamp_ns = static_cast<std::int64_t>(std::stoll(c[0]));
        sample.filename = c[1];
        out.push_back(sample);
    }
    return out;
}

std::vector<ImuSample> DatasetReader::loadImuCsvIfPresent(const std::string& path) {
    std::ifstream f(path.c_str());
    if (!f) return std::vector<ImuSample>();

    std::vector<ImuSample> out;
    std::string line;
    bool first_data_line = true;

    while (std::getline(f, line)) {
        if (isCommentOrBlank(line)) continue;
        const std::vector<std::string> c = splitCsv(line);
        if (first_data_line) {
            first_data_line = false;
            if (!c.empty() && c[0].find("timestamp") != std::string::npos) continue;
        }
        if (c.size() < 7) throw std::runtime_error("Bad imu.csv row: " + line);

        ImuSample sample;
        sample.timestamp_ns = static_cast<std::int64_t>(std::stoll(c[0]));
        sample.gyro_x = std::stod(c[1]);
        sample.gyro_y = std::stod(c[2]);
        sample.gyro_z = std::stod(c[3]);
        sample.accel_x = std::stod(c[4]);
        sample.accel_y = std::stod(c[5]);
        sample.accel_z = std::stod(c[6]);
        out.push_back(sample);
    }
    return out;
}

void DatasetReader::validateCamera(const std::vector<CameraSample>& camera) {
    if (camera.size() < 2) {
        throw std::runtime_error("Dataset needs at least two camera frames");
    }
    for (std::size_t i = 1; i < camera.size(); ++i) {
        if (camera[i].timestamp_ns <= camera[i - 1].timestamp_ns) {
            throw std::runtime_error("camera.csv timestamps are not strictly increasing");
        }
    }
}

void DatasetReader::validateImu(const std::vector<ImuSample>& imu) {
    for (std::size_t i = 1; i < imu.size(); ++i) {
        if (imu[i].timestamp_ns <= imu[i - 1].timestamp_ns) {
            throw std::runtime_error("imu.csv timestamps are not strictly increasing");
        }
    }
}

std::vector<RawImuSample> DatasetReader::loadRawCsv(const std::string& path) {
    std::ifstream f(path.c_str());
    std::vector<RawImuSample> out;
    std::string line;
    while (std::getline(f, line)) {
        if (isCommentOrBlank(line)) continue;
        const auto c = splitCsv(line);
        if (!c.empty() && c[0] == "timestamp_ns") continue;
        if (c.size() < 5) throw std::runtime_error("Bad raw IMU row: " + line);
        RawImuSample sample;
        sample.timestamp_ns = std::stoll(c[0]);
        sample.host_ts = std::stod(c[1]);
        sample.x = std::stod(c[2]); sample.y = std::stod(c[3]); sample.z = std::stod(c[4]);
        if (!out.empty() && sample.timestamp_ns <= out.back().timestamp_ns)
            throw std::runtime_error(path + " timestamps are not strictly increasing");
        out.push_back(sample);
    }
    return out;
}
