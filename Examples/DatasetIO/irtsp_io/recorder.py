"""Native DatasetIO writer: source seconds retained beside integer nanoseconds."""
import bisect
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
from .gst_video import rgb_png


def utc():
    return datetime.now(timezone.utc).isoformat()


class DatasetRecorder:
    def __init__(self, output, name=None, frame_capacity=120, timestamp_resolver=None):
        self.timestamp_resolver = timestamp_resolver
        self.output = Path(output).expanduser()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'camera').mkdir()
        self.files = {}
        self.writers = {}
        fields = {
            'camera': ['timestamp_ns', 'filename', 'host_ts', 'frame_index', 'width', 'height', 'pc_decode_ts', 'intrinsics_host_ts', 'intrinsics_delta_s', 'rtp_timestamp', 'ssrc'],
            'gyro': ['timestamp_ns', 'host_ts', 'x', 'y', 'z', 'seq', 'gap', 'pc_arrival_ts'],
            'accel': ['timestamp_ns', 'host_ts', 'x', 'y', 'z', 'seq', 'gap', 'pc_arrival_ts'],
            'intrinsics': ['timestamp_ns', 'host_ts', 'fx', 'fy', 'cx', 'cy', 'width', 'height', 'lens_position', 'focus_mode', 'adjusting_focus', 'exposure_duration', 'iso', 'seq', 'pc_arrival_ts'],
            'rtcp_sender_reports': ['ssrc', 'ntp_seconds', 'ntp_fraction', 'rtp_timestamp', 'packet_count', 'octet_count', 'receipt_pc_mono_s'],
        }
        for kind, columns in fields.items():
            f = (self.output / (kind + '.csv')).open('w', newline='', buffering=1)
            self.files[kind] = f
            self.writers[kind] = csv.DictWriter(f, fieldnames=columns)
            self.writers[kind].writeheader()
        self.metadata = dict(format='DatasetIO-raw-v1', session_name=name, start_utc=utc(),
                             timestamp_clock='iPhone monotonic host_ts; unre-based',
                             gyro_units='rad/s', accel_units='m/s^2', nominal_video_fps=30,
                             nominal_imu_rate_hz=100, irtsp_mode='raw', errors=[])
        self.times = {k: [] for k in ('camera', 'gyro', 'accel', 'intrinsics')}
        self.integrity = {k: dict(regressions=0, duplicates=0) for k in self.times}
        self.reports = []
        self.intrinsics = []
        self.pending = []
        self.missing = 0
        self.drops = 0
        self.written_frames = 0
        self.sequence_gaps = 0
        self.queue = queue.Queue(frame_capacity)
        self.worker = threading.Thread(target=self._write_images, name='dataset-png-writer')
        self.worker.start()
        self._save_metadata()

    def _save_metadata(self):
        tmp = self.output / 'session_info.json.tmp'
        tmp.write_text(json.dumps(self.metadata, indent=2, allow_nan=False))
        tmp.replace(self.output / 'session_info.json')

    def _track(self, kind, ts):
        if not math.isfinite(ts):
            raise RuntimeError(f'Invalid {kind} source timestamp')
        times = self.times[kind]
        if times:
            self.integrity[kind]['regressions'] += ts < times[-1]
            self.integrity[kind]['duplicates'] += ts == times[-1]
        times.append(ts)

    def consume(self, event, clock):
        import irtsp
        kind, now, data = event
        if kind == 'sr':
            self.reports.append(data)
            self.reports = self.reports[-64:]
            self.writers['rtcp_sender_reports'].writerow(data)
        elif kind == 'record':
            self.sequence_gaps += data.gap
            if isinstance(data, (irtsp.RawGyro, irtsp.RawAccel)):
                sensor = 'gyro' if isinstance(data, irtsp.RawGyro) else 'accel'
                vector = data.gyro if sensor == 'gyro' else data.accel
                self._track(sensor, data.host_ts)
                self.writers[sensor].writerow(dict(timestamp_ns=round(data.host_ts * 1e9), host_ts=data.host_ts,
                    x=vector.x, y=vector.y, z=vector.z, seq=data.seq, gap=data.gap, pc_arrival_ts=now))
            elif isinstance(data, irtsp.Intrinsics):
                self._track('intrinsics', data.host_ts)
                row = {k: getattr(data, k, None) for k in self.writers['intrinsics'].fieldnames}
                row.update(timestamp_ns=round(data.host_ts * 1e9), pc_arrival_ts=now)
                # Missing / non-finite optional fields remain explicitly unknown.
                row = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
                self.writers['intrinsics'].writerow(row)
                pos = bisect.bisect_right([r[0] for r in self.intrinsics], data.host_ts)
                self.intrinsics.insert(pos, (data.host_ts, data.seq, row))
        elif kind == 'frame':
            self.pending.append(data)
        elif kind == 'error':
            raise RuntimeError(str(data))
        self.resolve(clock)
        # Bounded source history; metadata CSV retains every original record.
        if self.pending:
            oldest = min(f['decode_delivery_pc_mono_s'] for f in self.pending)
            if now - oldest > 10:
                raise RuntimeError('No authoritative RTCP clock / matching intrinsics within 10 seconds')
        if len(self.pending) > 300:
            raise RuntimeError('Unresolved video buffer full; no clock or intrinsics association')
        if len(self.intrinsics) > 1800:
            self.intrinsics = self.intrinsics[-1800:]

    def resolve(self, clock, final=False):
        while self.pending:
            frame = self.pending[0]
            ts = (self.timestamp_resolver(frame, self.reports, clock) if self.timestamp_resolver
                  else frame.get('source_host_ts'))
            match = None
            if ts is not None:
                pos = bisect.bisect_left(self.intrinsics, (ts, -1))
                candidates = self.intrinsics[max(0, pos-1):pos+1]
                if candidates:
                    candidate = min(candidates, key=lambda r: abs(r[0]-ts))
                    if abs(candidate[0]-ts) <= 12e-6:
                        match = candidate[2]
            if match is None and not final:
                break
            self.pending.pop(0)
            if match is None:
                self.missing += 1
                self.metadata['errors'].append('Frame lacks authoritative clock/intrinsics association')
                # Retain inspectable orphan images; never invent their timestamps.
                frame['filename'] = f"camera/orphan_{frame['frame_index']:06d}.png"
            else:
                if (frame['width'], frame['height']) != (match['width'], match['height']):
                    raise RuntimeError('iRTSP frame/intrinsics geometry mismatch. Restart iRTSP while holding the phone in landscape orientation.')
                self._track('camera', ts)
                frame['filename'] = f"camera/{frame['frame_index']:06d}.png"
                frame['camera_row'] = dict(timestamp_ns=round(ts*1e9), filename=frame['filename'], host_ts=ts,
                    frame_index=frame['frame_index'], width=frame['width'], height=frame['height'],
                    pc_decode_ts=frame['decode_delivery_pc_mono_s'], intrinsics_host_ts=match['host_ts'],
                    intrinsics_delta_s=match['host_ts']-ts, rtp_timestamp=frame['rtp_timestamp'], ssrc=frame['ssrc'])
                self.metadata['stream_width'] = frame['width']
                self.metadata['stream_height'] = frame['height']
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                self.drops += 1
                raise RuntimeError('PNG writer queue full; frame lost')

    def _write_images(self):
        while True:
            frame = self.queue.get()
            try:
                if frame is None:
                    return
                png = rgb_png(frame['width'], frame['height'], frame['stride'], frame['pixels'])
                (self.output / frame['filename']).write_bytes(png)
                if 'camera_row' in frame:
                    self.writers['camera'].writerow(frame['camera_row'])
                    self.written_frames += 1
            except Exception as exc:
                self.metadata['errors'].append(f'Image writer: {exc}')
            finally:
                self.queue.task_done()

    def finish(self, clock=None, source=None):
        if clock:
            try:
                self.resolve(clock, final=True)
            except Exception as exc:
                self.metadata['errors'].append(str(exc))
        self.queue.put(None)
        self.worker.join()
        for f in self.files.values():
            f.flush()
            f.close()
        self.metadata.update(end_utc=utc(), timestamp_integrity=self.integrity,
            missing_intrinsics_frame_associations=self.missing, dropped_writer_frames=self.drops,
            source_sequence_gaps=self.sequence_gaps, dropped_acquisition_events=source.drops if source else {})
        for kind, times in self.times.items():
            self.metadata[kind + '_count'] = len(times)
            self.metadata['observed_' + kind + '_rate_hz'] = ((len(times)-1)/(times[-1]-times[0])
                if len(times)>1 and times[-1]>times[0] and not any(self.integrity[kind].values()) else None)
        self.metadata['camera_acquired_count'] = self.metadata['camera_count']
        self.metadata['camera_count'] = self.written_frames
        self.metadata['unresolved_frames'] = len(self.pending)
        if any(any(counts.values()) for counts in self.integrity.values()):
            self.metadata['errors'].append('Source timestamp regressions or duplicates detected')
        if self.written_frames < 2:
            self.metadata['errors'].append('Fewer than two recorded frames; session cannot be replayed')
        self.metadata['complete'] = not self.metadata['errors'] and not self.sequence_gaps and not self.missing and not self.pending and not self.drops and not any(source.drops.values() if source else [])
        self._save_metadata()
        return self.metadata
