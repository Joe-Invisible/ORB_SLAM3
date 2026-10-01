# FYP0852 Project Context

## Objective

Develop and evaluate a modern visual SLAM system suitable for navigation
through dynamic, previously unknown environments in real time.

The project began with a literature review of visual SLAM and dynamic-scene
SLAM. ORB-SLAM3 was selected as the principal baseline because of its mature
feature-based architecture and support for visual-inertial operation.

## Research direction

Dynamic-scene approaches reviewed include semantic, geometric, and hybrid
methods such as DynaSLAM, DS-SLAM, SegGeo-SLAM, RTS-SLAM, RigidFusion and
DynaVINS.

A recurring limitation of semantic-only rejection is that objects belonging
to potentially dynamic classes may be rejected even while stationary, while
pure geometric approaches have their own observability and initialization
limitations.

The project has therefore not committed to a specific dynamic-SLAM extension
yet. Current priority is establishing a robust working acquisition and
ORB-SLAM3 pipeline before adding dynamic-scene handling.

## Current engineering work

The system has progressed from evaluation on public datasets to live data
acquisition.

Current components include:

- live monocular ORB-SLAM3 operation
- camera calibration
- phone IMU acquisition and synchronization
- IMU noise characterization / Allan variance analysis
- visual-inertial experiments
- map / point-cloud export
- offline dataset recording and replay

Dataset recording is intentionally being separated from SLAM execution so
that the same capture can later be replayed through visual-only,
visual-inertial, and future algorithms.

The live runner is being generalized so sensor modes can eventually include
monocular, monocular-inertial, stereo and RGB-D.
