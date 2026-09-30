#pragma once

#include "LiveTypes.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// SensorLog UDP receiver. It preserves SensorLog's phone timestamps, rejects
// duplicate gyro measurements, and linearly interpolates accelerometer values
// onto each unique gyro timestamp, matching imu_sync_receiver.py.
class SensorLogImuReceiver {
public:
    explicit SensorLogImuReceiver(int port = 5555);
    ~SensorLogImuReceiver();

    SensorLogImuReceiver(const SensorLogImuReceiver&) = delete;
    SensorLogImuReceiver& operator=(const SensorLogImuReceiver&) = delete;

    bool start();
    void stop();

    // Wait for a short clock-mapping warm-up and at least one synchronized IMU
    // measurement. The host/phone offset is frozen after the warm-up so the
    // live camera timeline cannot jump later in the run.
    bool waitUntilReady(int timeout_ms);

    // Convert a host steady_clock time into SensorLog's seconds-since-reboot
    // clock using the frozen low-latency UDP clock offset.
    bool phoneTimeForHostNs(std::int64_t host_time_ns,
                            double& phone_time_s) const;

    // Wait until the receiver has produced a synchronized IMU sample strictly
    // after this phone timestamp. ORB-SLAM3's preintegrator benefits from one
    // sample beyond each camera boundary so it can interpolate to the exact
    // frame time instead of extrapolating from an older sample.
    bool waitUntilPhoneTime(double phone_time_s, int timeout_ms);

    void discardThrough(double phone_time_s);

    // Pop all synchronized samples through end_phone_time_s plus the first
    // sample strictly after it. That final sample is consumed here only once;
    // ORB-SLAM3 keeps it in its own IMU queue as the interpolation endpoint for
    // this frame and the starting-side sample for the next interval.
    void popThroughAndOneAfter(double end_phone_time_s,
                               double origin_phone_time_s,
                               std::vector<LiveImuSample>& output);

    std::size_t packetCount() const;
    std::size_t synchronizedCount() const;
    std::size_t duplicateGyroCount() const;
    std::size_t discardedGyroCount() const;
    std::string errorMessage() const;

private:
    struct AccelSample {
        double t, x, y, z;
    };

    struct GyroSample {
        double t, x, y, z;
    };

    struct SyncedSample {
        double phone_time_s;
        double gx, gy, gz;
        double ax, ay, az;
    };

    void receiverLoop();
    bool handlePacket(const std::string& payload, double recv_host_s);
    void processPendingGyros();
    bool findAccelBracket(double timestamp,
                          AccelSample& before,
                          AccelSample& after) const;
    void setError(const std::string& message);

    static bool extractJsonNumber(const std::string& payload,
                                  const char* key,
                                  double& value);
    static double hostSteadySeconds();

    const int port_;
    int socket_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread worker_;

    // Raw synchronization state is receiver-thread-only.
    std::deque<AccelSample> accel_buffer_;
    std::deque<GyroSample> pending_gyros_;
    bool have_last_gyro_timestamp_ = false;
    double last_gyro_timestamp_ = 0.0;
    bool have_last_output_timestamp_ = false;
    double last_output_timestamp_ = 0.0;

    mutable std::mutex mutex_;
    std::condition_variable data_cv_;
    std::deque<SyncedSample> synced_samples_;

    bool have_clock_candidate_ = false;
    bool clock_frozen_ = false;
    double host_minus_phone_s_ = 0.0;
    std::size_t clock_observations_ = 0;

    bool have_latest_synced_time_ = false;
    double latest_synced_phone_time_s_ = 0.0;

    std::atomic<std::size_t> packet_count_{0};
    std::atomic<std::size_t> synced_count_{0};
    std::atomic<std::size_t> duplicate_gyro_count_{0};
    std::atomic<std::size_t> discarded_gyro_count_{0};
    std::string error_message_;
};
