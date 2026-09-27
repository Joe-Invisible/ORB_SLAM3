#include <System.h>
#include "ImuTypes.h"

#include "DatasetReader.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

namespace {

enum SensorMode {
    SENSOR_MONO,
    SENSOR_MONO_IMU
};

struct Args {
    std::string vocabulary;
    std::string settings;
    std::string dataset;
    std::string output_dir = "dataset_results";
    SensorMode sensor = SENSOR_MONO;
    bool realtime = true;
    bool viewer = true;
};

std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a[a.size() - 1] == '/') return a + b;
    return a + "/" + b;
}

void makeDirectories(const std::string& path) {
    if (path.empty()) return;
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
                throw std::runtime_error("mkdir(" + current + ") failed: " + std::strerror(errno));
            }
        }
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
}

Args parseArgs(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error(
            "Usage: orbslam_dataset_player VOCAB SETTINGS DATASET "
            "[--sensor mono|mono-imu] [--realtime|--fast] "
            "[--no-viewer] [--output-dir DIR]");
    }

    Args a;
    a.vocabulary = argv[1];
    a.settings = argv[2];
    a.dataset = argv[3];

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sensor") {
            if (++i >= argc) throw std::runtime_error("--sensor requires mono or mono-imu");
            const std::string mode = argv[i];
            if (mode == "mono") a.sensor = SENSOR_MONO;
            else if (mode == "mono-imu") a.sensor = SENSOR_MONO_IMU;
            else throw std::runtime_error("Unknown sensor mode: " + mode);
        } else if (arg == "--realtime") {
            a.realtime = true;
        } else if (arg == "--fast") {
            a.realtime = false;
        } else if (arg == "--no-viewer") {
            a.viewer = false;
        } else if (arg == "--output-dir") {
            if (++i >= argc) throw std::runtime_error("--output-dir requires a path");
            a.output_dir = argv[i];
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return a;
}

inline double secondsFrom(std::int64_t timestamp_ns, std::int64_t origin_ns) {
    return static_cast<double>(timestamp_ns - origin_ns) * 1e-9;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        const DatasetReader dataset(args.dataset);
        const std::vector<CameraSample>& camera = dataset.camera();
        const std::vector<ImuSample>& imu = dataset.imu();

        if (args.sensor == SENSOR_MONO_IMU && imu.empty()) {
            throw std::runtime_error(
                "--sensor mono-imu requested, but this dataset has no IMU stream");
        }

        const std::int64_t origin_ns =
            (args.sensor == SENSOR_MONO_IMU)
                ? std::min(camera.front().timestamp_ns, imu.front().timestamp_ns)
                : camera.front().timestamp_ns;

        std::cout << "Loaded " << camera.size() << " camera frames";
        if (!imu.empty()) std::cout << " and " << imu.size() << " IMU samples";
        std::cout << "\nSensor mode: "
                  << (args.sensor == SENSOR_MONO ? "monocular" : "monocular-inertial")
                  << "\nPlayback: " << (args.realtime ? "realtime" : "fast/unthrottled")
                  << "\nTimestamp origin: " << origin_ns << " ns\n";

        if (args.sensor == SENSOR_MONO && !imu.empty()) {
            std::cout << "IMU stream present but intentionally ignored in monocular mode.\n";
        }

        makeDirectories(args.output_dir);

        const ORB_SLAM3::System::eSensor orb_sensor =
            (args.sensor == SENSOR_MONO)
                ? ORB_SLAM3::System::MONOCULAR
                : ORB_SLAM3::System::IMU_MONOCULAR;

        ORB_SLAM3::System slam(args.vocabulary, args.settings, orb_sensor, args.viewer);

        std::size_t imu_index = 0;
        if (args.sensor == SENSOR_MONO_IMU) {
            if (imu.front().timestamp_ns > camera.front().timestamp_ns) {
                std::cerr << "WARNING: IMU begins after the first camera frame. "
                             "Visual-inertial initialization may be weakened.\n";
            }
            while (imu_index < imu.size() &&
                   imu[imu_index].timestamp_ns <= camera.front().timestamp_ns) {
                ++imu_index;
            }
            if (imu_index > 0) --imu_index;
        }

        std::vector<ORB_SLAM3::IMU::Point> imu_batch;
        double total_track_s = 0.0;

        for (std::size_t i = 0; i < camera.size(); ++i) {
            const std::string image_path = dataset.absolutePath(camera[i].filename);
            cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                throw std::runtime_error("Cannot load image " + image_path);
            }

            const double frame_time_s = secondsFrom(camera[i].timestamp_ns, origin_ns);
            imu_batch.clear();

            if (args.sensor == SENSOR_MONO_IMU && i > 0) {
                while (imu_index < imu.size() &&
                       imu[imu_index].timestamp_ns <= camera[i].timestamp_ns) {
                    const ImuSample& m = imu[imu_index];
                    imu_batch.push_back(ORB_SLAM3::IMU::Point(
                        m.accel_x, m.accel_y, m.accel_z,
                        m.gyro_x, m.gyro_y, m.gyro_z,
                        secondsFrom(m.timestamp_ns, origin_ns)));
                    ++imu_index;
                }
            }

            const std::chrono::steady_clock::time_point begin =
                std::chrono::steady_clock::now();

            if (args.sensor == SENSOR_MONO) {
                slam.TrackMonocular(image, frame_time_s);
            } else {
                slam.TrackMonocular(image, frame_time_s, imu_batch);
            }

            const std::chrono::steady_clock::time_point end =
                std::chrono::steady_clock::now();
            const double track_s =
                std::chrono::duration_cast<std::chrono::duration<double> >(end - begin).count();
            total_track_s += track_s;

            if ((i + 1) % 30 == 0 || i + 1 == camera.size()) {
                std::cout << "\rFrame " << (i + 1) << '/' << camera.size();
                if (args.sensor == SENSOR_MONO_IMU) {
                    std::cout << " | IMU batch " << imu_batch.size();
                }
                std::cout << " | track " << std::fixed << std::setprecision(1)
                          << (track_s * 1000.0) << " ms" << std::flush;
            }

            if (args.realtime && i + 1 < camera.size()) {
                const double frame_period_s =
                    static_cast<double>(camera[i + 1].timestamp_ns -
                                        camera[i].timestamp_ns) * 1e-9;
                if (track_s < frame_period_s) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(frame_period_s - track_s));
                }
            }
        }
        std::cout << '\n';

        slam.Shutdown();

        if (args.sensor == SENSOR_MONO) {
            // ORB-SLAM3's monocular examples conventionally save keyframes.
            slam.SaveKeyFrameTrajectoryTUM(
                joinPath(args.output_dir, "KeyFrameTrajectory.txt"));
        } else {
            slam.SaveTrajectoryEuRoC(
                joinPath(args.output_dir, "CameraTrajectory.txt"));
            slam.SaveKeyFrameTrajectoryEuRoC(
                joinPath(args.output_dir, "KeyFrameTrajectory.txt"));
        }

        std::cout << "Finished. Mean TrackMonocular time: "
                  << std::fixed << std::setprecision(2)
                  << (1000.0 * total_track_s / static_cast<double>(camera.size()))
                  << " ms\nSaved trajectories under: " << args.output_dir << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
