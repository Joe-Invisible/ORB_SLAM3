#include "LiveConfig.h"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace {
LiveSensorMode parseSensorMode(const std::string& s) {
    if (s == "mono") return LiveSensorMode::MONO;
    if (s == "mono-imu") return LiveSensorMode::MONO_IMU;
    if (s == "stereo") return LiveSensorMode::STEREO;
    if (s == "stereo-imu") return LiveSensorMode::STEREO_IMU;
    if (s == "rgbd") return LiveSensorMode::RGBD;
    throw std::runtime_error("Unknown sensor mode: " + s);
}

int parsePositiveInt(const std::string& text, const std::string& option) {
    const int value = std::atoi(text.c_str());
    if (value <= 0) {
        throw std::runtime_error(option + " requires a positive integer");
    }
    return value;
}
}

std::string LiveUsage() {
    std::ostringstream os;
    os
        << "Usage:\n"
        << "  live_runner --source rtsp --rtsp URL --sensor mono [actions/options]\n\n"
        << "Actions (enable at least one):\n"
        << "  --slam                     Run ORB-SLAM3\n"
        << "  --record DIR               Record the incoming sensor stream\n\n"
        << "SLAM options:\n"
        << "  --vocab FILE               ORB vocabulary (required with --slam)\n"
        << "  --settings FILE            ORB-SLAM3 settings (required with --slam)\n"
        << "  --viewer / --no-viewer     Enable/disable Pangolin viewer\n"
        << "  --output-dir DIR           SLAM outputs (default: live_results)\n\n"
        << "Input options:\n"
        << "  --source rtsp              Current implemented source\n"
        << "  --rtsp URL                 RTSP camera URL\n"
        << "  --sensor MODE              mono | mono-imu | stereo | stereo-imu | rgbd\n"
        << "  --width N                  Expected camera width (default: 1280)\n"
        << "  --height N                 Expected camera height (default: 720)\n\n"
        << "Current source limitation:\n"
        << "  The RTSP source supplies one camera only, so it currently supports\n"
        << "  --sensor mono. Other modes already have downstream data types/SLAM\n"
        << "  adapters, but need the corresponding acquisition source.\n";
    return os.str();
}

LiveConfig ParseLiveConfig(int argc, char** argv) {
    LiveConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            throw std::runtime_error(LiveUsage());
        } else if (arg == "--source") {
            if (++i >= argc) throw std::runtime_error("--source requires a value");
            cfg.source = argv[i];
        } else if (arg == "--rtsp") {
            if (++i >= argc) throw std::runtime_error("--rtsp requires a URL");
            cfg.rtsp_url = argv[i];
        } else if (arg == "--sensor") {
            if (++i >= argc) throw std::runtime_error("--sensor requires a mode");
            cfg.sensor = parseSensorMode(argv[i]);
        } else if (arg == "--slam") {
            cfg.run_slam = true;
        } else if (arg == "--record") {
            if (++i >= argc) throw std::runtime_error("--record requires a directory");
            cfg.record = true;
            cfg.dataset_dir = argv[i];
        } else if (arg == "--vocab") {
            if (++i >= argc) throw std::runtime_error("--vocab requires a file");
            cfg.vocabulary_path = argv[i];
        } else if (arg == "--settings") {
            if (++i >= argc) throw std::runtime_error("--settings requires a file");
            cfg.settings_path = argv[i];
        } else if (arg == "--viewer") {
            cfg.viewer = true;
        } else if (arg == "--no-viewer") {
            cfg.viewer = false;
        } else if (arg == "--output-dir") {
            if (++i >= argc) throw std::runtime_error("--output-dir requires a directory");
            cfg.output_dir = argv[i];
        } else if (arg == "--width") {
            if (++i >= argc) throw std::runtime_error("--width requires an integer");
            cfg.expected_width = parsePositiveInt(argv[i], "--width");
        } else if (arg == "--height") {
            if (++i >= argc) throw std::runtime_error("--height requires an integer");
            cfg.expected_height = parsePositiveInt(argv[i], "--height");
        } else {
            throw std::runtime_error("Unknown argument: " + arg + "\n\n" + LiveUsage());
        }
    }

    if (!cfg.run_slam && !cfg.record) {
        throw std::runtime_error(
            "Nothing to do: enable --slam and/or --record DIR\n\n" + LiveUsage());
    }

    if (cfg.source != "rtsp") {
        throw std::runtime_error("Unsupported source: " + cfg.source);
    }

    if (cfg.rtsp_url.empty()) {
        throw std::runtime_error("--rtsp URL is required for --source rtsp");
    }

    if (cfg.run_slam) {
        if (cfg.vocabulary_path.empty()) {
            throw std::runtime_error("--vocab FILE is required with --slam");
        }
        if (cfg.settings_path.empty()) {
            throw std::runtime_error("--settings FILE is required with --slam");
        }
    }

    return cfg;
}
