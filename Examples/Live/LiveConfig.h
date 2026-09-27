#pragma once

#include "LiveTypes.h"

#include <string>

struct LiveConfig {
    std::string source = "rtsp";
    LiveSensorMode sensor = LiveSensorMode::MONO;

    std::string rtsp_url;

    bool run_slam = false;
    bool record = false;
    bool viewer = true;

    std::string vocabulary_path;
    std::string settings_path;
    std::string dataset_dir;
    std::string output_dir = "live_results";

    int expected_width = 1280;
    int expected_height = 720;
};

LiveConfig ParseLiveConfig(int argc, char** argv);
std::string LiveUsage();
