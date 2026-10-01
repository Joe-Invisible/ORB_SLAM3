# iRTSP recording in native DatasetIO

Start iRTSP streaming with the phone held **landscape**, and select **RAW**
sensor mode. Starting portrait and turning the phone afterwards is insufficient.
The current validated configuration is 1280 × 720 at approximately 30 Hz with
independent raw gyro and accelerometer near 100 Hz.

From `/home/joe/slam/ORB_SLAM3`:

```bash
/home/joe/slam/imu_test/.venv-irtsp/bin/python Examples/DatasetIO/irtsp_recorder.py \
  --host "$PH" --output ~/slam/dataset/my_session --duration 30 \
  --session-name my_session
python3 Examples/DatasetIO/validate_irtsp_dataset.py ~/slam/dataset/my_session
```

Omit `--duration` (or use zero) to record until Ctrl+C. Existing output directories
are refused. There is deliberately no destructive overwrite option. `--session-name`
is descriptive metadata; `--output` selects the actual directory.

## Components and dependencies

`irtsp_io/source.py:IrtspSource` uses the official `irtsp` client for raw IMU
and camera metadata. It subscribes before opening to retain startup records.
`irtsp_io/gst_video.py:GstVideo` promotes the validated backend from
`~/slam/imu_test/irtsp_gst_video.py`; the original diagnostic files are intact.
It uses GStreamer RTSP/TCP, H264 decoding, and RTP/RTCP sender reports. It copies
RGB buffers without rotating, resizing or modifying K.

`irtsp_io/recorder.py:DatasetRecorder` owns native dataset writing and associations.
The source supplies a timestamp resolver; file writing does not open transport
connections. The source event queue is bounded (8192 events), as are the PNG
writer queue (120 frames), unresolved frames (300 / ten seconds), intrinsics
history (1800), and sender-report history (64). Queue overflow is counted and
terminates capture visibly. PNG compression runs on a separate worker, while
the official IMU reader continues receiving records. CSV rows preserve each
stream's acquisition order. No gyro/accel pairing or interpolation occurs.

The existing Python 3.10 venv uses `irtsp==0.8.0.post1`. System GStreamer,
GI (`python3-gi`, Gst/GstRtp/GstSdp/GstVideo bindings), and rtspsrc,
rtph264depay, h264parse, avdec_h264, videoconvert, capsfilter and appsink are
required. The backend exposes the existing `/usr/lib/python3/dist-packages`
GI installation to the venv. PyAV must remain unloaded in this process because
its bundled FFmpeg conflicts with this system GStreamer/libav installation.

## Files and clock semantics

- `camera/*.png`: lossless original RGB pixels.
- `camera.csv`: native `timestamp_ns,filename` first, then original host seconds,
  frame index, dimensions, PC decode time, matched intrinsics time/delta, RTP
  timestamp and SSRC. Only successfully written images receive camera rows.
- `gyro.csv`, `accel.csv`: independent `timestamp_ns,host_ts,x,y,z,seq,gap,pc_arrival_ts`.
  Gyroscope units are **rad/s**; acceleration units are **m/s²**. Values and axes
  are exactly those delivered by the official raw API; no frame transformation
  or gravity subtraction is applied.
- `intrinsics.csv`: every received record's host timestamp, fx/fy/cx/cy, dimensions,
  optional focus/exposure/ISO fields and sequence/receipt diagnostics. Missing
  optional API fields remain blank. The pinned client does not expose the
  advertised focus fields, so blanks mean unknown, never locked or settled.
- `rtcp_sender_reports.csv`: integer RTP/NTP evidence used for video mapping.
- `session_info.json`: UTC start/end, source configuration, clock anchors,
  backend/software versions, counts/rates, geometry, integrity/drop counters,
  errors and completion state. Nominal rates are project expectations; observed
  rates come from source timestamps.

All authoritative timestamps use the iPhone monotonic host clock, without
rebasing. Original double seconds are serialized using Python's round-trip
representation; integer nanoseconds are `round(host_ts * 1e9)`. Video host time
is reconstructed from RTCP sender-report NTP, RTP ticks, and the phone's
advertised host/wall anchors. PC timestamps are diagnostics only. No arrival
clock fitting, stream-delay compensation or timestamp adjustment to K occurs.
90 kHz RTP quantization means video timestamps may differ from corresponding
intrinsics by a few microseconds. Matching tolerance is 12 µs, separately stored
as a delta; it is not described as bit-exact matching.

Every associated frame must have identical decoded and intrinsics dimensions.
Mismatch aborts with a landscape restart instruction; images and K are never
silently corrected. Missing RTCP or intrinsics associations wait up to ten
seconds. At shutdown unassociated frames become `camera/orphan_*.png`, with no
invented camera CSV timestamp; metadata marks the session incomplete.

Duration expiry and Ctrl+C stop acquisition, drain received events, finish PNG
writes, flush CSVs, and atomically finalize metadata. Failed sessions retain
written files and report errors. Process kill/power loss cannot drain in-memory
queues; existing CSV/PNG files remain inspectable, with initial metadata rather
than a finalized completion report. Inspect integrity counters before using data.
Global odometry sequence gaps are distinct from application queue losses;
per-sensor sequence gaps cannot be interpreted as sensor drops. The implementation
does not claim packet-level RTP loss auditing; the validator detects source-time
gaps. Timestamp history used for rates currently grows with session length.

## Existing offline pipeline

This extends the existing format, rather than introducing another camera dataset.
The C++ `DatasetReader` reads camera timestamps/filenames directly and now exposes
`rawGyro()` and `rawAccel()` independently. Existing combined `imu.csv` remains
supported, and is never synthesized during raw recording.

`orbslam_dataset_player ... --sensor mono --no-viewer` consumes these camera files
unchanged, using a separately calibrated fixed camera YAML. Reader readback was
tested; running ORB-SLAM on the black validation images was intentionally omitted.
`--sensor mono-imu` reports that an explicit resampling adapter is needed for
raw-only datasets. That adapter, IMU::Point conversion, camera/IMU extrinsics,
and Kalibr conversion are future work. All their raw input data is retained.

The recorder is ready to collect camera calibration, camera/IMU calibration, and
ordinary SLAM datasets. Useful calibration requires a visible calibration target;
visual-inertial calibration requires appropriate motion and later conversion.
SLAM testing additionally requires a calibrated camera model, and inertial SLAM
requires the future replay adapter and valid calibration.

## Validation on 2026-10-01

The initial 25-second physical-phone recording at
`~/slam/dataset/irtsp_validation_20261001` contained 749 frames, 2531 gyro samples,
2531 accel samples, and 763 intrinsics records. Source rates were 29.9994 Hz and
99.5573 Hz. Every PNG decoded as 1280 × 720, and every frame matched intrinsics
(maximum absolute delta 7.398 µs). There were zero regressions, duplicates, gaps
over 1.5 expected periods, missing associations, queue drops or odometry sequence
gaps. Extra intrinsics at the capture boundaries are retained intentionally.
The phone faced down on a table, so these black images validate storage and
synchronization, not scene quality or SLAM tracking.

The player target builds successfully (existing upstream warnings). Standalone
C++ DatasetReader readback recovered all camera and independent raw IMU counts.
Run hardware-independent failure tests with:

```bash
/home/joe/slam/imu_test/.venv-irtsp/bin/python Examples/DatasetIO/test_irtsp_recorder.py
```

Final capture after source/recorder separation and integrity improvements:
`~/slam/dataset/irtsp_validation_final_20261001`, 25 seconds, 750 camera frames,
2526 gyro and 2526 accel samples, 761 intrinsics records; source rates 29.9994 Hz
and 99.5564 Hz. All 750 PNGs fully decoded at 1280 × 720. Maximum timestamp
association delta 10.686 µs; all integrity, missing association and drop counters
were zero, with no gaps above 1.5 nominal periods. Nine hardware-independent tests
cover geometry rejection, independent timestamps, queue overflows, draining,
unmapped frame preservation, monotonicity, overwrite refusal and injected Ctrl+C
finalization. Ctrl+C was injected in a test; duration expiry was exercised on the
physical phone. The metadata version-reporting addition and empty-session guard
were verified by unit tests after this capture.
