#!/usr/bin/env python3
"""Record iRTSP camera and independent raw IMU streams in native DatasetIO format."""
import argparse
from dataclasses import asdict
import importlib.metadata
import math
import queue
import platform
import subprocess
from pathlib import Path
import time
from irtsp_io.source import IrtspSource
from irtsp_io.recorder import DatasetRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=0, help='seconds; 0 until Ctrl+C')
    parser.add_argument('--session-name')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('duration must be finite and nonnegative')
    recorder = DatasetRecorder(args.output, args.session_name, timestamp_resolver=IrtspSource.timestamp)
    source = IrtspSource(args.host)
    clock = None
    try:
        source.open()
        clock = source.phone.clock
        recorder.metadata.update(source_host=args.host, software=dict(irtsp=importlib.metadata.version('irtsp'), python=platform.python_version(),
                                 recorder_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                                 recorder_worktree='See repository diff; revision may include uncommitted recorder changes'),
                                 clock_anchor=asdict(clock), video_backend=source.video.audit,
                                 source_configuration={k:v for k,v in source.phone.info.raw.items() if k in ('mode', 'rate_hz', 'capture_settings', 'imu_units', 'raw_accel_units', 'raw_gyro_units', 'intrinsics_units', 'version', 'protocol')})
        start = last = time.monotonic()
        print(f'Recording to {recorder.output}; use landscape stream startup.', flush=True)
        while not source.stop.is_set():
            now = time.monotonic()
            if args.duration and now-start >= args.duration:
                break
            try:
                recorder.consume(source.events.get(timeout=.1), clock)
            except queue.Empty:
                pass
            if now-start > 10 and any(not recorder.times[k] for k in ('camera', 'gyro', 'accel')):
                raise RuntimeError('No camera/raw gyro/raw accel within 10 seconds; check stream and RAW mode')
            if source.phone.closed:
                raise RuntimeError('iRTSP odometry connection closed')
            if recorder.metadata['errors']:
                raise RuntimeError(recorder.metadata['errors'][-1])
            if now-last >= 5:
                print({k: len(v) for k,v in recorder.times.items()}, flush=True)
                last = now
    except KeyboardInterrupt:
        print('Stopping and draining recording.', flush=True)
    except Exception as exc:
        recorder.metadata['errors'].append(str(exc))
        print(f'ERROR: {exc}', flush=True)
    finally:
        source.close()
        if source.error:
            recorder.metadata['errors'].append(source.error)
        while not source.events.empty():
            event = source.events.get_nowait()
            if clock:
                try:
                    recorder.consume(event, clock)
                except Exception as exc:
                    recorder.metadata['errors'].append(str(exc))
        summary = recorder.finish(clock, source)
        print({k:v for k,v in summary.items() if k.endswith('_count') or k.startswith('observed_') or k in ('complete','errors','dropped_acquisition_events','dropped_writer_frames','missing_intrinsics_frame_associations')}, flush=True)
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
