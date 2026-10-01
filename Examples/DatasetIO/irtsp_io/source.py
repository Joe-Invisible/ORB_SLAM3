"""Reusable iRTSP acquisition. No filesystem or SLAM dependencies."""
import queue
import threading
import time
from .gst_video import GstVideo, rtcp_host


class IrtspSource:
    def __init__(self, host, capacity=8192):
        self.host = host
        self.events = queue.Queue(capacity)
        self.stop = threading.Event()
        self.drops = {}
        self.error = None
        self.phone = None
        self.worker = None

    def emit(self, kind, now, data):
        # Packet ingress is used internally by GstVideo; retaining every packet
        # in the recorder queue would compete with measurements needlessly.
        if kind == 'rtp':
            return
        try:
            self.events.put_nowait((kind, now, data))
        except queue.Full:
            self.drops[kind] = self.drops.get(kind, 0) + 1
            self.error = 'Acquisition queue full; data lost'
            self.stop.set()

    def open(self):
        import irtsp
        # Subscribe before opening: avoids losing the replayed initial metadata.
        self.phone = irtsp.Session(self.host, reconnect=False, timeout=5)
        self.phone.on(irtsp.Record, lambda rec: self.emit('record', time.monotonic(), rec))
        self.phone.open()
        if self.phone.info.raw.get('mode') != 'raw':
            self.phone.close()
            raise RuntimeError('Enable RAW sensor mode in iRTSP before recording')
        self.video = GstVideo(self.phone.video_url, self.stop, self.emit, snapshots_every=1)
        self.worker = threading.Thread(target=self._video, name='irtsp-video')
        self.worker.start()
        return self

    @staticmethod
    def timestamp(frame, reports, clock):
        return rtcp_host(frame["rtp_timestamp"], frame["ssrc"], reports, clock)

    def _video(self):
        try:
            for frame in self.video:
                self.emit('frame', time.monotonic(), frame)
            if not self.stop.is_set():
                self.error = 'Video stream ended'
                self.stop.set()
        except Exception as exc:
            self.error = str(exc)
            self.stop.set()

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=8)
            if self.worker.is_alive():
                self.video.pipeline.set_state(self.video.Gst.State.NULL)
                self.worker.join(timeout=3)
                self.error = self.error or 'Video shutdown timed out'
        if self.phone:
            self.phone.close()
