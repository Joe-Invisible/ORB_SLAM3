#pragma once

#include "LiveTypes.h"

#include <string>

// Acquisition-only interface. It deliberately has no ORB-SLAM3 dependency.
class LiveSource {
public:
    virtual ~LiveSource() {}

    virtual bool open() = 0;
    virtual bool read(LiveFrame& frame) = 0;
    virtual void close() = 0;

    virtual bool supports(LiveSensorMode mode) const = 0;
    virtual std::string description() const = 0;
};
