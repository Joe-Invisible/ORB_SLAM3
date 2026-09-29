#include "SensorLogImuReceiver.h"

#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
constexpr double kGravity = 9.80665;
constexpr double kTimestampEpsilonS = 1e-7;
constexpr std::size_t kAccelBufferSize = 20;
constexpr std::size_t kClockWarmupObservations = 50;
}

SensorLogImuReceiver::SensorLogImuReceiver(int port) : port_(port) {}
SensorLogImuReceiver::~SensorLogImuReceiver() { stop(); }

double SensorLogImuReceiver::hostSteadySeconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool SensorLogImuReceiver::extractJsonNumber(const std::string& payload,
                                             const char* key,
                                             double& value) {
    const std::string needle = std::string("\"") + key + "\"";
    const std::size_t key_pos = payload.find(needle);
    if (key_pos == std::string::npos) return false;

    const std::size_t colon = payload.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return false;

    const char* p = payload.c_str() + colon + 1;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    if (*p == '"') ++p;

    char* end = nullptr;
    errno = 0;
    value = std::strtod(p, &end);
    return errno == 0 && end != p && std::isfinite(value);
}

bool SensorLogImuReceiver::start() {
    if (running_) return true;

    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        setError(std::string("socket() failed: ") + std::strerror(errno));
        return false;
    }

    const int reuse = 1;
    ::setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<unsigned short>(port_));

    if (::bind(socket_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        setError(std::string("bind UDP port ") + std::to_string(port_) +
                 " failed: " + std::strerror(errno));
        ::close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }

    timeval timeout{};
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    ::setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    running_ = true;
    worker_ = std::thread(&SensorLogImuReceiver::receiverLoop, this);
    return true;
}

void SensorLogImuReceiver::stop() {
    running_ = false;
    data_cv_.notify_all();

    if (worker_.joinable()) worker_.join();

    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

void SensorLogImuReceiver::receiverLoop() {
    char buffer[65536];

    while (running_) {
        const ssize_t n = ::recvfrom(socket_fd_, buffer, sizeof(buffer) - 1, 0,
                                     nullptr, nullptr);
        if (n < 0) {
            if (!running_) break;
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
            setError(std::string("recvfrom() failed: ") + std::strerror(errno));
            break;
        }

        const double recv_host_s = hostSteadySeconds();
        buffer[n] = '\0';
        handlePacket(std::string(buffer, static_cast<std::size_t>(n)), recv_host_s);
    }
}

bool SensorLogImuReceiver::handlePacket(const std::string& payload,
                                        double recv_host_s) {
    double accel_t, ax, ay, az, gyro_t, gx, gy, gz;

    if (!extractJsonNumber(payload, "accelerometerTimestamp_sinceReboot", accel_t) ||
        !extractJsonNumber(payload, "accelerometerAccelerationX", ax) ||
        !extractJsonNumber(payload, "accelerometerAccelerationY", ay) ||
        !extractJsonNumber(payload, "accelerometerAccelerationZ", az) ||
        !extractJsonNumber(payload, "gyroTimestamp_sinceReboot", gyro_t) ||
        !extractJsonNumber(payload, "gyroRotationX", gx) ||
        !extractJsonNumber(payload, "gyroRotationY", gy) ||
        !extractJsonNumber(payload, "gyroRotationZ", gz)) {
        std::cerr << "WARNING: SensorLog UDP packet missing an expected numeric field\n";
        return false;
    }

    ++packet_count_;

    if (accel_buffer_.empty() ||
        accel_t > accel_buffer_.back().t + kTimestampEpsilonS) {
        accel_buffer_.push_back({accel_t, ax * kGravity, ay * kGravity, az * kGravity});
        while (accel_buffer_.size() > kAccelBufferSize) accel_buffer_.pop_front();
    }

    const bool gyro_is_new = !have_last_gyro_timestamp_ ||
        gyro_t > last_gyro_timestamp_ + kTimestampEpsilonS;

    if (gyro_is_new) {
        // Arrival time = phone sample time + clock offset + network/processing
        // delay. During a short warm-up, the minimum observed difference is a
        // robust estimate of the offset because queueing only makes arrivals
        // later. Freeze it before camera timestamps are generated.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!clock_frozen_) {
                const double candidate = recv_host_s - gyro_t;
                if (!have_clock_candidate_ || candidate < host_minus_phone_s_) {
                    host_minus_phone_s_ = candidate;
                    have_clock_candidate_ = true;
                }
                ++clock_observations_;
                if (have_clock_candidate_ &&
                    clock_observations_ >= kClockWarmupObservations) {
                    clock_frozen_ = true;
                    data_cv_.notify_all();
                }
            }
        }

        pending_gyros_.push_back({gyro_t, gx, gy, gz});
        last_gyro_timestamp_ = gyro_t;
        have_last_gyro_timestamp_ = true;
    } else {
        ++duplicate_gyro_count_;
    }

    processPendingGyros();
    return true;
}

bool SensorLogImuReceiver::findAccelBracket(double timestamp,
                                            AccelSample& before,
                                            AccelSample& after) const {
    if (accel_buffer_.size() < 2) return false;

    for (std::size_t i = 0; i + 1 < accel_buffer_.size(); ++i) {
        if (accel_buffer_[i].t <= timestamp &&
            timestamp <= accel_buffer_[i + 1].t) {
            before = accel_buffer_[i];
            after = accel_buffer_[i + 1];
            return true;
        }
    }
    return false;
}

void SensorLogImuReceiver::processPendingGyros() {
    while (!pending_gyros_.empty()) {
        const GyroSample gyro = pending_gyros_.front();

        if (!accel_buffer_.empty() && gyro.t < accel_buffer_.front().t) {
            pending_gyros_.pop_front();
            ++discarded_gyro_count_;
            continue;
        }

        if (accel_buffer_.empty() || accel_buffer_.back().t < gyro.t) break;

        AccelSample a0{}, a1{};
        if (!findAccelBracket(gyro.t, a0, a1)) {
            pending_gyros_.pop_front();
            ++discarded_gyro_count_;
            continue;
        }

        const double dt = a1.t - a0.t;
        const double alpha = std::abs(dt) < kTimestampEpsilonS
            ? 0.0 : (gyro.t - a0.t) / dt;

        const double ax = a0.x + alpha * (a1.x - a0.x);
        const double ay = a0.y + alpha * (a1.y - a0.y);
        const double az = a0.z + alpha * (a1.z - a0.z);

        pending_gyros_.pop_front();

        if (have_last_output_timestamp_ && gyro.t <= last_output_timestamp_) {
            ++discarded_gyro_count_;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            synced_samples_.push_back({gyro.t, gyro.x, gyro.y, gyro.z, ax, ay, az});
            latest_synced_phone_time_s_ = gyro.t;
            have_latest_synced_time_ = true;
        }

        last_output_timestamp_ = gyro.t;
        have_last_output_timestamp_ = true;
        ++synced_count_;
        data_cv_.notify_all();
    }
}

bool SensorLogImuReceiver::waitUntilReady(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return !running_ ||
               (clock_frozen_ && have_latest_synced_time_ && !synced_samples_.empty());
    });

    return clock_frozen_ && have_latest_synced_time_ && !synced_samples_.empty();
}

bool SensorLogImuReceiver::phoneTimeForHostNs(std::int64_t host_time_ns,
                                              double& phone_time_s) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!clock_frozen_) return false;

    phone_time_s = static_cast<double>(host_time_ns) * 1e-9 - host_minus_phone_s_;
    return true;
}

bool SensorLogImuReceiver::waitUntilPhoneTime(double phone_time_s,
                                              int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return !running_ ||
               (have_latest_synced_time_ &&
                latest_synced_phone_time_s_ >= phone_time_s);
    });

    return have_latest_synced_time_ &&
           latest_synced_phone_time_s_ >= phone_time_s;
}

void SensorLogImuReceiver::discardThrough(double phone_time_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!synced_samples_.empty() &&
           synced_samples_.front().phone_time_s <= phone_time_s) {
        synced_samples_.pop_front();
    }
}

void SensorLogImuReceiver::popThrough(double end_phone_time_s,
                                      double origin_phone_time_s,
                                      std::vector<LiveImuSample>& output) {
    output.clear();
    std::lock_guard<std::mutex> lock(mutex_);

    while (!synced_samples_.empty() &&
           synced_samples_.front().phone_time_s <= end_phone_time_s) {
        const SyncedSample s = synced_samples_.front();
        synced_samples_.pop_front();

        LiveImuSample m;
        m.timestamp_ns = static_cast<std::int64_t>(
            std::llround((s.phone_time_s - origin_phone_time_s) * 1e9));
        m.gyro_x = s.gx;
        m.gyro_y = s.gy;
        m.gyro_z = s.gz;
        m.accel_x = s.ax;
        m.accel_y = s.ay;
        m.accel_z = s.az;
        output.push_back(m);
    }
}

std::size_t SensorLogImuReceiver::packetCount() const { return packet_count_.load(); }
std::size_t SensorLogImuReceiver::synchronizedCount() const { return synced_count_.load(); }
std::size_t SensorLogImuReceiver::duplicateGyroCount() const { return duplicate_gyro_count_.load(); }
std::size_t SensorLogImuReceiver::discardedGyroCount() const { return discarded_gyro_count_.load(); }

std::string SensorLogImuReceiver::errorMessage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_message_;
}

void SensorLogImuReceiver::setError(const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error_message_ = message;
    }
    std::cerr << "ERROR: SensorLog IMU receiver: " << message << '\n';
    running_ = false;
    data_cv_.notify_all();
}
