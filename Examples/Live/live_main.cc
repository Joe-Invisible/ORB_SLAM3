#include "LiveConfig.h"
#include "LiveSource.h"
#include "OrbSlamRunner.h"
#include "RtspCameraSource.h"
#include "RtspMonoImuSource.h"

#include "DatasetRecorder.h"

#include <csignal>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
volatile std::sig_atomic_t g_running = 1;

void signalHandler(int) {
    g_running = 0;
}

std::unique_ptr<LiveSource> makeSource(const LiveConfig& cfg) {
    if (cfg.source == "rtsp") {
        if (cfg.sensor == LiveSensorMode::MONO_IMU) {
            return std::unique_ptr<LiveSource>(new RtspMonoImuSource(
                cfg.rtsp_url,
                cfg.expected_width,
                cfg.expected_height,
                cfg.imu_port,
                cfg.imu_timeout_ms,
                cfg.camera_time_offset_ms));
        }

        return std::unique_ptr<LiveSource>(new RtspCameraSource(
            cfg.rtsp_url,
            cfg.expected_width,
            cfg.expected_height));
    }

    throw std::runtime_error("No implementation for source: " + cfg.source);
}
}

int main(int argc, char** argv) {
    try {
        const LiveConfig cfg = ParseLiveConfig(argc, argv);

        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);

        std::unique_ptr<LiveSource> source = makeSource(cfg);
        if (!source->supports(cfg.sensor)) {
            throw std::runtime_error(
                source->description() + " does not provide sensor mode '" +
                LiveSensorModeName(cfg.sensor) + "'. Add/select a source that supplies "
                "the required streams.");
        }

        std::cout
            << "\n----------------------------------------\n"
            << "ORB-SLAM3 Generic Live Frontend\n"
            << "----------------------------------------\n"
            << "Source : " << source->description() << '\n'
            << "Sensor : " << LiveSensorModeName(cfg.sensor) << '\n'
            << "SLAM   : " << (cfg.run_slam ? "enabled" : "disabled") << '\n'
            << "Record : " << (cfg.record ? cfg.dataset_dir : "disabled") << "\n\n";

        // Deliberately initialise acquisition before ORB-SLAM3. Record-only mode
        // therefore never constructs a SLAM system at all.
        if (!source->open()) return 1;

        std::unique_ptr<DatasetRecorder> recorder;
        std::ofstream camera_timing_csv;
        if (cfg.record) {
            recorder.reset(new DatasetRecorder(cfg.dataset_dir));
            if (!recorder->good()) {
                throw std::runtime_error(recorder->errorMessage());
            }
            std::cout << "Dataset recording enabled: " << cfg.dataset_dir << '\n';
        }

        std::unique_ptr<OrbSlamRunner> slam;
        if (cfg.run_slam) {
            slam.reset(new OrbSlamRunner(
                cfg.sensor,
                cfg.vocabulary_path,
                cfg.settings_path,
                cfg.viewer,
                cfg.output_dir));
        }

        std::cout << "Live processing started. Press Ctrl+C to stop.\n\n";

        std::size_t captured_frames = 0;
        LiveFrame frame;

        while (g_running) {
            if (!source->read(frame)) break;

            // Recording and SLAM consume the same LiveFrame and timestamp.
            if (recorder) {
                if (!recorder->recordFrame(frame.timestamp_ns, frame.image0)) {
                    throw std::runtime_error(recorder->errorMessage());
                }

                // Source-specific timing diagnostics stay out of the canonical
                // camera.csv format. Open the sidecar lazily so sources that
                // do not provide these diagnostics create no extra file.
                if (frame.camera_timing.valid) {
                    if (!camera_timing_csv.is_open()) {
                        camera_timing_csv.open(
                            (cfg.dataset_dir + "/camera_timing.csv").c_str(),
                            std::ios::out | std::ios::trunc);
                        if (!camera_timing_csv) {
                            throw std::runtime_error(
                                "Failed to open camera_timing.csv in " + cfg.dataset_dir);
                        }
                        camera_timing_csv
                            << "timestamp_ns,source_pts_ns,source_dts_ns,"
                            << "host_pull_ns,host_copy_done_ns,"
                            << "host_mapped_phone_ns,frame_phone_time_ns\n";
                    }
                    const LiveCameraTiming& t = frame.camera_timing;
                    camera_timing_csv
                        << frame.timestamp_ns << ',' << t.source_pts_ns << ','
                        << t.source_dts_ns << ',' << t.host_pull_ns << ','
                        << t.host_copy_done_ns << ',' << t.host_mapped_phone_ns
                        << ',' << t.frame_phone_time_ns << '\n';
                }

                for (std::size_t i = 0; i < frame.imu.size(); ++i) {
                    const LiveImuSample& m = frame.imu[i];
                    if (!recorder->recordImu(
                            m.timestamp_ns,
                            m.gyro_x, m.gyro_y, m.gyro_z,
                            m.accel_x, m.accel_y, m.accel_z)) {
                        throw std::runtime_error(recorder->errorMessage());
                    }
                }
            }

            if (slam && !slam->process(frame)) {
                throw std::runtime_error("ORB-SLAM3 rejected a live frame");
            }

            ++captured_frames;

            if (!slam && captured_frames % 30 == 0) {
                std::cout
                    << "\rRecorded/captured frames: " << captured_frames
                    << " | t: " << std::fixed << std::setprecision(2)
                    << (static_cast<double>(frame.timestamp_ns) * 1e-9) << " s"
                    << "       " << std::flush;
            }
        }

        if (camera_timing_csv.is_open()) {
            camera_timing_csv.flush();
            if (!camera_timing_csv) {
                throw std::runtime_error("Failed while writing camera_timing.csv");
            }
            camera_timing_csv.close();
        }

        source->close();

        if (recorder) {
            std::cout << "\nFlushing dataset recorder...\n";
            if (!recorder->finish()) {
                throw std::runtime_error(recorder->errorMessage());
            }
            std::cout
                << "Dataset saved: " << cfg.dataset_dir
                << " | camera frames=" << recorder->frameCount()
                << " | IMU samples=" << recorder->imuCount() << '\n';
        }

        if (slam) slam->finish();

        std::cout << "Live frontend stopped.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << '\n';
        return 1;
    }
}
