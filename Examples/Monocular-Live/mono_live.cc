#include <System.h>
#include <Tracking.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

namespace
{
    volatile std::sig_atomic_t running = 1;

    // -------------------------------------------------------------------------
    // Logging thresholds
    // -------------------------------------------------------------------------

    constexpr double SLOW_TRACKING_THRESHOLD_MS = 100.0;

    // A tracking interruption is considered significant if either:
    //   - it lasts at least 0.1 seconds, OR
    //   - it persists for at least 3 processed frames.
    constexpr double SIGNIFICANT_LOSS_THRESHOLD_S = 0.1;
    constexpr size_t SIGNIFICANT_LOSS_MIN_FRAMES = 3;

    // -------------------------------------------------------------------------
    // Signal handling
    // -------------------------------------------------------------------------

    void SignalHandler(int)
    {
        running = 0;
    }

    // -------------------------------------------------------------------------
    // Convert ORB-SLAM3 tracking state to readable text
    // -------------------------------------------------------------------------

    const char* TrackingStateName(int state)
    {
        switch (state)
        {
            case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
                return "SYSTEM_NOT_READY";

            case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                return "NO_IMAGES_YET";

            case ORB_SLAM3::Tracking::NOT_INITIALIZED:
                return "NOT_INITIALIZED";

            case ORB_SLAM3::Tracking::OK:
                return "OK";

            case ORB_SLAM3::Tracking::RECENTLY_LOST:
                return "RECENTLY_LOST";

            case ORB_SLAM3::Tracking::LOST:
                return "LOST";

            case ORB_SLAM3::Tracking::OK_KLT:
                return "OK_KLT";

            default:
                return "UNKNOWN";
        }
    }
}

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Command-line arguments
    // -------------------------------------------------------------------------

    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./mono_live "
             << "path_to_vocabulary "
             << "path_to_settings "
             << "rtsp_url"
             << endl;

        return 1;
    }

    const string vocabularyPath = argv[1];
    const string settingsPath   = argv[2];
    const string rtspUrl        = argv[3];

    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);

    cout << endl
         << "----------------------------------------" << endl
         << "ORB-SLAM3 Live Monocular" << endl
         << "----------------------------------------" << endl
         << "RTSP stream: " << rtspUrl << endl
         << endl;

    // -------------------------------------------------------------------------
    // Initialise ORB-SLAM3
    // -------------------------------------------------------------------------

    ORB_SLAM3::System SLAM(
        vocabularyPath,
        settingsPath,
        ORB_SLAM3::System::MONOCULAR,
        true
    );

    // -------------------------------------------------------------------------
    // Low-latency GStreamer pipeline
    // -------------------------------------------------------------------------

    const string pipeline =
        "rtspsrc location=\"" + rtspUrl + "\" "
        "protocols=udp "
        "latency=50 "
        "drop-on-latency=true ! "
        "rtph264depay ! "
        "h264parse ! "
        "avdec_h264 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false";

    cout << "Opening RTSP stream..." << endl;

    cv::VideoCapture cap(
        pipeline,
        cv::CAP_GSTREAMER
    );

    if (!cap.isOpened())
    {
        cerr << "ERROR: Could not open RTSP stream."
             << endl;

        SLAM.Shutdown();

        return 1;
    }

    cout << "RTSP stream opened." << endl;
    cout << "Waiting for first frame..." << endl;

    cv::Mat frame;

    if (!cap.read(frame) || frame.empty())
    {
        cerr << "ERROR: Could not receive first frame."
             << endl;

        cap.release();
        SLAM.Shutdown();

        return 1;
    }

    // -------------------------------------------------------------------------
    // Verify calibration resolution
    // -------------------------------------------------------------------------

    if (frame.cols != 1280 || frame.rows != 720)
    {
        cerr << "ERROR: Unexpected stream resolution: "
             << frame.cols << "x" << frame.rows << endl
             << "Expected 1280x720." << endl
             << "The camera calibration would therefore be invalid."
             << endl;

        cap.release();
        SLAM.Shutdown();

        return 1;
    }

    cout << "First frame received: "
         << frame.cols << "x" << frame.rows
         << endl;

    // -------------------------------------------------------------------------
    // Statistics CSV
    // -------------------------------------------------------------------------

    ofstream statsFile(
        "live_tracking_stats.csv"
    );

    if (!statsFile.is_open())
    {
        cerr << "ERROR: Could not create "
             << "live_tracking_stats.csv"
             << endl;

        cap.release();
        SLAM.Shutdown();

        return 1;
    }

    statsFile
        << "frame,"
        << "timestamp_s,"
        << "frame_dt_ms,"
        << "state,"
        << "track_ms,"
        << "slow_track,"
        << "keypoints,"
        << "map_point_associations"
        << '\n';

    cout << endl
         << "Live tracking started." << endl
         << "Press Ctrl+C to stop." << endl
         << "Statistics: live_tracking_stats.csv" << endl
         << "Slow-frame threshold: "
         << SLOW_TRACKING_THRESHOLD_MS
         << " ms" << endl
         << "Significant-loss threshold: "
         << SIGNIFICANT_LOSS_THRESHOLD_S
         << " s or "
         << SIGNIFICANT_LOSS_MIN_FRAMES
         << " frames"
         << endl
         << endl;

    // -------------------------------------------------------------------------
    // Timing/statistics state
    // -------------------------------------------------------------------------

    const auto startTime =
        chrono::steady_clock::now();

    size_t frameCount = 0;

    vector<double> trackingTimesMs;
    trackingTimesMs.reserve(10000);

    size_t slowTrackingFrames = 0;

    // Any transition from normal OK tracking into
    // RECENTLY_LOST or LOST.
    size_t trackingInterruptions = 0;

    // Interruptions satisfying the duration/frame threshold.
    size_t significantLossEpisodes = 0;

    // Interruptions shorter than the threshold.
    size_t briefInterruptions = 0;

    // Loss episodes that returned to OK without resetting.
    size_t recoveredEpisodes = 0;

    // Loss episodes that resulted in reinitialisation/reset.
    size_t failedRecoveryEpisodes = 0;

    bool lossEpisodeActive = false;

    double lossStartTimestamp = 0.0;
    size_t lossStartFrame = 0;

    int previousState =
        ORB_SLAM3::Tracking::NO_IMAGES_YET;

    double previousTimestamp = 0.0;

    // -------------------------------------------------------------------------
    // Main processing loop
    // -------------------------------------------------------------------------

    while (running)
    {
        // The first frame was acquired before entering the loop.
        // Every later iteration fetches the newest available frame.
        if (frameCount > 0)
        {
            if (!cap.read(frame))
            {
                cerr << endl
                     << "WARNING: Failed to retrieve frame."
                     << endl;

                break;
            }

            if (frame.empty())
            {
                cerr << endl
                     << "WARNING: Received empty frame."
                     << endl;

                continue;
            }
        }

        // ---------------------------------------------------------------------
        // Timestamp
        // ---------------------------------------------------------------------

        const auto frameTime =
            chrono::steady_clock::now();

        const double timestamp =
            chrono::duration<double>(
                frameTime - startTime
            ).count();

        const double frameDtMs =
            (frameCount == 0)
                ? 0.0
                : (timestamp - previousTimestamp)
                    * 1000.0;

        // ---------------------------------------------------------------------
        // ORB-SLAM3 tracking
        // ---------------------------------------------------------------------

        const auto trackStart =
            chrono::steady_clock::now();

        SLAM.TrackMonocular(
            frame,
            timestamp
        );

        const auto trackEnd =
            chrono::steady_clock::now();

        const double trackingTimeMs =
            chrono::duration<double, milli>(
                trackEnd - trackStart
            ).count();

        trackingTimesMs.push_back(
            trackingTimeMs
        );

        // ---------------------------------------------------------------------
        // Diagnostics for the frame just processed
        // ---------------------------------------------------------------------

        const int state =
            SLAM.GetTrackingState();

        const auto keypoints =
            SLAM.GetTrackedKeyPointsUn();

        const auto mapPoints =
            SLAM.GetTrackedMapPoints();

        const size_t validMapPoints =
            count_if(
                mapPoints.begin(),
                mapPoints.end(),
                [](ORB_SLAM3::MapPoint* pMP)
                {
                    return pMP != nullptr;
                }
            );

        // ---------------------------------------------------------------------
        // Slow TrackMonocular() warning
        // ---------------------------------------------------------------------

        const bool slowTracking =
            trackingTimeMs >
            SLOW_TRACKING_THRESHOLD_MS;

        if (slowTracking)
        {
            ++slowTrackingFrames;

            cout << endl
                 << "[SLOW TRACKING]"
                 << " frame=" << frameCount
                 << " t="
                 << fixed << setprecision(3)
                 << timestamp << " s"
                 << " state="
                 << TrackingStateName(state)
                 << " TrackMonocular="
                 << setprecision(1)
                 << trackingTimeMs << " ms"
                 << endl;
        }

        // ---------------------------------------------------------------------
        // Print state transition
        // ---------------------------------------------------------------------

        if (state != previousState)
        {
            cout << endl
                 << "[STATE] "
                 << TrackingStateName(previousState)
                 << " -> "
                 << TrackingStateName(state)
                 << " at t="
                 << fixed << setprecision(3)
                 << timestamp << " s"
                 << endl;
        }

        // ---------------------------------------------------------------------
        // Begin a tracking interruption
        //
        // Normally:
        //
        //     OK -> RECENTLY_LOST
        //
        // but count a direct OK -> LOST as well.
        // ---------------------------------------------------------------------

        if (!lossEpisodeActive &&
            previousState == ORB_SLAM3::Tracking::OK &&
            (state == ORB_SLAM3::Tracking::RECENTLY_LOST ||
             state == ORB_SLAM3::Tracking::LOST))
        {
            lossEpisodeActive = true;

            lossStartTimestamp = timestamp;
            lossStartFrame = frameCount;

            ++trackingInterruptions;

            cout << "[LOSS] Tracking interruption "
                 << trackingInterruptions
                 << " began at t="
                 << fixed << setprecision(3)
                 << timestamp << " s"
                 << endl;
        }

        // ---------------------------------------------------------------------
        // Successful recovery
        //
        // Existing tracking recovered directly to OK.
        // ---------------------------------------------------------------------

        if (lossEpisodeActive &&
            state == ORB_SLAM3::Tracking::OK)
        {
            const double lossDuration =
                timestamp - lossStartTimestamp;

            const size_t lostFrames =
                frameCount - lossStartFrame;

            const bool significant =
                lossDuration >=
                    SIGNIFICANT_LOSS_THRESHOLD_S ||
                lostFrames >=
                    SIGNIFICANT_LOSS_MIN_FRAMES;

            if (significant)
            {
                ++significantLossEpisodes;
            }
            else
            {
                ++briefInterruptions;
            }

            ++recoveredEpisodes;

            lossEpisodeActive = false;

            cout << "[RECOVERY] Tracking recovered at t="
                 << fixed << setprecision(3)
                 << timestamp << " s"
                 << " | duration="
                 << lossDuration << " s"
                 << " | lost_frames="
                 << lostFrames
                 << " | "
                 << (
                        significant
                            ? "SIGNIFICANT"
                            : "BRIEF"
                    )
                 << endl;
        }

        // ---------------------------------------------------------------------
        // Failed recovery / reinitialisation
        //
        // If an active loss episode falls back into an initialization state,
        // then normal tracking was not recovered.
        // ---------------------------------------------------------------------

        if (lossEpisodeActive &&
            (state ==
                ORB_SLAM3::Tracking::NO_IMAGES_YET ||
             state ==
                ORB_SLAM3::Tracking::NOT_INITIALIZED))
        {
            const double lossDuration =
                timestamp - lossStartTimestamp;

            const size_t lostFrames =
                frameCount - lossStartFrame;

            // A reset/reinitialisation is always treated as
            // a significant tracking failure.
            ++significantLossEpisodes;
            ++failedRecoveryEpisodes;

            lossEpisodeActive = false;

            cout << "[REINITIALISATION] "
                 << "Tracking recovery failed at t="
                 << fixed << setprecision(3)
                 << timestamp << " s"
                 << " | duration="
                 << lossDuration << " s"
                 << " | lost_frames="
                 << lostFrames
                 << endl;
        }

        // ---------------------------------------------------------------------
        // Write per-frame statistics
        // ---------------------------------------------------------------------

        statsFile
            << frameCount << ','
            << fixed << setprecision(6)
            << timestamp << ','
            << setprecision(3)
            << frameDtMs << ','
            << TrackingStateName(state) << ','
            << trackingTimeMs << ','
            << (slowTracking ? 1 : 0) << ','
            << keypoints.size() << ','
            << validMapPoints
            << '\n';

        // Make sure abnormal events are written to disk immediately.
        if (slowTracking)
        {
            statsFile.flush();
        }

        // ---------------------------------------------------------------------
        // Advance frame state
        // ---------------------------------------------------------------------

        ++frameCount;

        previousState = state;
        previousTimestamp = timestamp;

        // ---------------------------------------------------------------------
        // Live terminal status
        //
        // Roughly once per second when operating near 30 fps.
        // ---------------------------------------------------------------------

        if (frameCount % 30 == 0)
        {
            const double meanTrackingTime =
                accumulate(
                    trackingTimesMs.begin(),
                    trackingTimesMs.end(),
                    0.0
                ) /
                static_cast<double>(
                    trackingTimesMs.size()
                );

            cout << "\r"
                 << "Frames: "
                 << frameCount

                 << " | t: "
                 << fixed << setprecision(2)
                 << timestamp << " s"

                 << " | state: "
                 << TrackingStateName(state)

                 << " | track: "
                 << setprecision(1)
                 << trackingTimeMs << " ms"

                 << " | mean: "
                 << meanTrackingTime << " ms"

                 << " | kp: "
                 << keypoints.size()

                 << " | MPs: "
                 << validMapPoints

                 << "       "
                 << flush;
        }
    }

    // -------------------------------------------------------------------------
    // Shutdown
    // -------------------------------------------------------------------------

    cout << endl
         << endl
         << "Stopping live SLAM..."
         << endl;

    cap.release();

    SLAM.Shutdown();

    statsFile.close();

    // -------------------------------------------------------------------------
    // Save trajectory
    // -------------------------------------------------------------------------

    cout << "Saving keyframe trajectory..."
         << endl;

    SLAM.SaveKeyFrameTrajectoryTUM(
        "KeyFrameTrajectory.txt"
    );

    // -------------------------------------------------------------------------
    // Save sparse point clouds
    //
    // One PLY is produced for each valid map in the Atlas:
    //
    //     ORB_SLAM3_PointCloud_map_0.ply
    //     ORB_SLAM3_PointCloud_map_1.ply
    //     ...
    //
    // Requires SavePointCloudPLY() to have been added to System.h/System.cc.
    // -------------------------------------------------------------------------

    cout << "Saving sparse point cloud..."
         << endl;

    SLAM.SavePointCloudPLY(
        "ORB_SLAM3_PointCloud"
    );

    // -------------------------------------------------------------------------
    // Final timing statistics
    // -------------------------------------------------------------------------

    if (!trackingTimesMs.empty())
    {
        vector<double> sortedTimes =
            trackingTimesMs;

        sort(
            sortedTimes.begin(),
            sortedTimes.end()
        );

        const size_t n =
            sortedTimes.size();

        const double mean =
            accumulate(
                sortedTimes.begin(),
                sortedTimes.end(),
                0.0
            ) /
            static_cast<double>(n);

        double variance = 0.0;

        for (const double t : sortedTimes)
        {
            const double d =
                t - mean;

            variance +=
                d * d;
        }

        variance /=
            static_cast<double>(n);

        const double stddev =
            sqrt(variance);

        // Median
        double median = 0.0;

        if (n % 2 == 0)
        {
            median =
                (
                    sortedTimes[n / 2 - 1] +
                    sortedTimes[n / 2]
                ) / 2.0;
        }
        else
        {
            median =
                sortedTimes[n / 2];
        }

        // 95th percentile
        const size_t p95Index =
            min(
                static_cast<size_t>(
                    ceil(
                        0.95 *
                        static_cast<double>(n)
                    )
                ) - 1,
                n - 1
            );

        const double p95 =
            sortedTimes[p95Index];

        const double minTime =
            sortedTimes.front();

        const double maxTime =
            sortedTimes.back();

        const double elapsed =
            previousTimestamp;

        const double processedFPS =
            (
                elapsed > 0.0 &&
                frameCount > 1
            )
                ? static_cast<double>(
                      frameCount - 1
                  ) / elapsed
                : 0.0;

        const double slowFramePercentage =
            100.0 *
            static_cast<double>(
                slowTrackingFrames
            ) /
            static_cast<double>(n);

        // ---------------------------------------------------------------------
        // Summary
        // ---------------------------------------------------------------------

        cout << endl
             << "========== Live SLAM Statistics =========="
             << endl

             << "Frames processed : "
             << frameCount << endl

             << "Elapsed time     : "
             << fixed << setprecision(2)
             << elapsed << " s" << endl

             << "Processed FPS    : "
             << processedFPS << endl

             << endl

             << "Tracking time:" << endl

             << "  mean           : "
             << mean << " ms" << endl

             << "  median         : "
             << median << " ms" << endl

             << "  std dev        : "
             << stddev << " ms" << endl

             << "  min            : "
             << minTime << " ms" << endl

             << "  max            : "
             << maxTime << " ms" << endl

             << "  95th percentile: "
             << p95 << " ms" << endl

             << endl

             << "Slow frames > "
             << SLOW_TRACKING_THRESHOLD_MS
             << " ms : "
             << slowTrackingFrames
             << " ("
             << slowFramePercentage
             << "%)"
             << endl

             << endl

             << "Tracking interruptions : "
             << trackingInterruptions
             << endl

             << "Significant losses     : "
             << significantLossEpisodes
             << endl

             << "Brief interruptions    : "
             << briefInterruptions
             << endl

             << "Recovered episodes     : "
             << recoveredEpisodes
             << endl

             << "Failed recoveries      : "
             << failedRecoveryEpisodes
             << endl

             << "Unresolved at shutdown : "
             << (lossEpisodeActive ? 1 : 0)
             << endl

             << "=========================================="
             << endl;
    }

    cout << endl
         << "Trajectory saved to "
         << "KeyFrameTrajectory.txt"
         << endl

         << "Tracking statistics saved to "
         << "live_tracking_stats.csv"
         << endl

         << "Sparse point clouds saved as "
         << "ORB_SLAM3_PointCloud_map_*.ply"
         << endl;

    return 0;
}
