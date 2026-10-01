"""GStreamer RTCP/RTP audit backend. No clock fitting or latency correction."""

import sys
import time
import struct
import zlib


def rgb_png(width, height, stride, data):
    """Encode a captured RGB buffer verbatim; no rotation or geometry changes."""
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff)
    scanlines = b''.join(b'\0' + bytes(data[y * stride:y * stride + width * 3]) for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scanlines, 1)) + chunk(b'IEND', b''))


def load_gst():
    # Ubuntu's GI bindings are already installed for the same Python 3.10 ABI.
    # Make only that existing package visible to the isolated project venv.
    try:
        import gi
    except ImportError:
        sys.path.append('/usr/lib/python3/dist-packages')
        import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtp', '1.0')
    gi.require_version('GstSdp', '1.0')
    from gi.repository import Gst, GstRtp, GstSdp
    # Register SDP boxed types before rtspsrc emits on-sdp (otherwise GBoxed).
    _ = GstSdp.SDPMessage
    Gst.init(None)
    return Gst, GstRtp


def rtcp_host(timestamp, ssrc, reports, clock):
    candidates = [r for r in reports if r['ssrc'] == ssrc]
    if not candidates or timestamp is None:
        return None
    # Use the SR with the closest RTP timestamp, including wrap-around.
    def delta(report):
        return ((timestamp - report['rtp_timestamp'] + 2**31) % 2**32) - 2**31
    report = min(candidates, key=lambda r: abs(delta(r)))
    # Avoid subtracting two large Unix floats before recovering host time.
    # Integer NTP seconds/fraction are retained in the CSV for independent audit.
    host_at_sr = clock.host_anchor + ((report['ntp_seconds'] - 2208988800)
                                      - clock.wall_anchor) + report['ntp_fraction'] / 2**32
    return host_at_sr + delta(report) / 90000


class GstVideo:
    """Decode frames and keep RTP ingress, post-jitter PTS and SR evidence."""

    def __init__(self, url, stop, emit, snapshots_every=0):
        Gst, GstRtp = load_gst()
        self.Gst, self.GstRtp = Gst, GstRtp
        self.stop, self.emit = stop, emit
        self.pipeline = Gst.Pipeline.new('irtsp-source-clock-audit')
        self.source = Gst.ElementFactory.make('rtspsrc', 'source')
        self.depay = Gst.ElementFactory.make('rtph264depay', 'depay')
        self.parser = Gst.ElementFactory.make('h264parse', 'parser')
        self.decoder = Gst.ElementFactory.make('avdec_h264', 'decoder')
        self.sink = Gst.ElementFactory.make('appsink', 'sink')
        if not all((self.source, self.depay, self.parser, self.decoder, self.sink)):
            raise RuntimeError('Missing GStreamer rtspsrc/H264/appsink plugin')
        self.source.set_property('location', url)
        self.source.set_property('protocols', 4)  # TCP; matches official Python video transport.
        # Deliberately leave latency at its default; measure, do not compensate.
        self.sink.set_property('sync', False)
        self.sink.set_property('max-buffers', 32)
        self.sink.set_property('drop', False)
        elements = [self.source, self.depay, self.parser, self.decoder]
        if snapshots_every:
            converter = Gst.ElementFactory.make('videoconvert', 'rgb_converter')
            filter_ = Gst.ElementFactory.make('capsfilter', 'rgb_caps')
            filter_.set_property('caps', Gst.Caps.from_string('video/x-raw,format=RGB'))
            elements.extend([converter, filter_])
        elements.append(self.sink)
        for element in elements:
            self.pipeline.add(element)
        for a, b in zip(elements[1:], elements[2:]):
            if not a.link(b):
                raise RuntimeError(f'Cannot link {a.name} to {b.name}')
        self.source.connect('pad-added', self._source_pad)
        self.source.connect('new-manager', self._manager)
        self.source.connect('on-sdp', lambda source, sdp: self.emit('configuration', time.monotonic(),
                                                                  dict(stage='sdp', details=sdp.as_text())))
        self.pts_to_rtp = {}
        self.receipts = {}
        self.decoder_times = {}
        self.snapshots_every = snapshots_every
        self.last_snapshot = -float('inf')
        self.encoded_size = (None, None)
        self.configuration = {}
        for name, element in (('encoded', self.parser), ('decoded', self.decoder)):
            element.get_static_pad('src').add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM,
                                                   self._configuration_event, name)
        self.decoder.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, self._decoded)
        self.audit = {'gstreamer_version': Gst.version_string(),
                      'rtspsrc_latency_ms': self.source.get_property('latency'),
                      'video_network_boundary': 'rtpbin recv_rtp_sink ingress after TCP deframing, before jitterbuffer',
                      'decode_boundary': 'avdec_h264 src pad (before appsink pull); no RGB conversion/display',
                      'mapping': 'decoded buffer PTS -> post-jitter RTP PTS -> RTP timestamp -> RTCP SR NTP -> odometry host anchor'}

    def _configuration_event(self, pad, info, stage):
        event = info.get_event()
        details = None
        if event.type == self.Gst.EventType.CAPS:
            caps = event.parse_caps()
            details = caps.to_string()
            if stage == 'encoded':
                structure = caps.get_structure(0)
                self.encoded_size = (structure.get_value('width'), structure.get_value('height'))
        elif event.type == self.Gst.EventType.TAG:
            details = event.parse_tag().to_string()
        if details is not None:
            key = (stage, str(event.type))
            if self.configuration.get(key) != details:
                self.configuration[key] = details
                self.emit('configuration', time.monotonic(), dict(stage=stage, event=str(event.type), details=details))
        return self.Gst.PadProbeReturn.OK

    def _decoded(self, pad, info):
        self.decoder_times[info.get_buffer().pts] = time.monotonic()
        return self.Gst.PadProbeReturn.OK

    def _rtp(self, buffer):
        ok, rtp = self.GstRtp.RTPBuffer.map(buffer, self.Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            return dict(ssrc=rtp.get_ssrc(), rtp_timestamp=rtp.get_timestamp(),
                        rtp_seq=rtp.get_seq(), marker=rtp.get_marker())
        finally:
            rtp.unmap()

    def _ingress(self, pad, info):
        now = time.monotonic()
        record = self._rtp(info.get_buffer())
        if record:
            key = (record['ssrc'], record['rtp_timestamp'])
            first, last = self.receipts.get(key, (now, now))
            self.receipts[key] = (first, now)
            record['ingress_pc_mono_s'] = now
            self.emit('rtp', now, record)
        return self.Gst.PadProbeReturn.OK

    def _rtcp(self, pad, info):
        now = time.monotonic()
        rtcp = self.GstRtp.RTCPBuffer()
        if self.GstRtp.RTCPBuffer.map(info.get_buffer(), self.Gst.MapFlags.READ, rtcp):
            try:
                packet = self.GstRtp.RTCPPacket()
                if rtcp.get_first_packet(packet):
                    while True:
                        if packet.get_type() == self.GstRtp.RTCPType.SR:
                            ssrc, ntp, rtp, count, octets = packet.sr_get_sender_info()
                            self.emit('sr', now, dict(ssrc=ssrc, ntp_seconds=ntp >> 32,
                                                     ntp_fraction=ntp & 0xffffffff,
                                                     rtp_timestamp=rtp, packet_count=count,
                                                     octet_count=octets, receipt_pc_mono_s=now))
                        if not packet.move_to_next():
                            break
            finally:
                rtcp.unmap()
        return self.Gst.PadProbeReturn.OK

    def _manager(self, source, manager):
        def attach(manager, pad):
            if pad.name.startswith('recv_rtp_sink_'):
                pad.add_probe(self.Gst.PadProbeType.BUFFER, self._ingress)
            elif pad.name.startswith('recv_rtcp_sink_'):
                pad.add_probe(self.Gst.PadProbeType.BUFFER, self._rtcp)
        manager.connect('pad-added', attach)
        for pad in manager.pads:
            attach(manager, pad)

    def _source_pad(self, source, pad):
        caps = pad.get_current_caps()
        if not caps:
            return
        structure = caps.get_structure(0)
        if structure.get_string('media') != 'video':
            return
        self.emit('configuration', time.monotonic(), dict(stage='rtp', details=caps.to_string()))
        if structure.get_string('encoding-name') != 'H264':
            self.emit('error', time.monotonic(), 'GStreamer backend currently requires H264 video')
            return
        if structure.get_value('clock-rate') != 90000:
            self.emit('error', time.monotonic(), 'Unexpected video RTP clock rate (requires 90000)')
            return

        def post_jitter(pad, info):
            buffer = info.get_buffer()
            record = self._rtp(buffer)
            if record and buffer.pts != self.Gst.CLOCK_TIME_NONE:
                self.pts_to_rtp[buffer.pts] = (record['ssrc'], record['rtp_timestamp'])
            return self.Gst.PadProbeReturn.OK
        pad.add_probe(self.Gst.PadProbeType.BUFFER, post_jitter)
        pad.link(self.depay.get_static_pad('sink'))

    def __iter__(self):
        Gst = self.Gst
        self.pipeline.set_state(Gst.State.PLAYING)
        index = 0
        bus = self.pipeline.get_bus()
        try:
            while not self.stop.is_set():
                sample = self.sink.emit('try-pull-sample', 100_000_000)
                if sample:
                    delivered = time.monotonic()
                    buffer = sample.get_buffer()
                    caps = sample.get_caps().get_structure(0)
                    ssrc, rtp = self.pts_to_rtp.get(buffer.pts, (None, None))
                    first, last = self.receipts.get((ssrc, rtp), (None, None))
                    decoded = self.decoder_times.pop(buffer.pts, delivered)
                    import gi
                    gi.require_version('GstVideo', '1.0')
                    from gi.repository import GstVideo
                    video_info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
                    ok, mapping = buffer.map(Gst.MapFlags.READ)
                    if not ok:
                        raise RuntimeError('Cannot map decoded RGB frame')
                    try:
                        pixels = bytes(mapping.data)
                    finally:
                        buffer.unmap(mapping)
                    self.pts_to_rtp.pop(buffer.pts, None)
                    self.receipts.pop((ssrc, rtp), None)
                    yield dict(pixels=pixels, stride=video_info.stride[0], frame_index=index, pts_s=buffer.pts / 1e9,
                               api_unix_ts=None, approx_clock=None, source_host_ts=None,
                               decode_delivery_pc_mono_s=decoded, appsink_pull_pc_mono_s=delivered,
                               network_receipt_pc_mono_s=first, rtp_last_ingress_pc_mono_s=last,
                               rtp_timestamp=rtp, ssrc=ssrc,
                               width=caps.get_value('width'), height=caps.get_value('height'),
                               encoded_width=self.encoded_size[0], encoded_height=self.encoded_size[1])
                    index += 1
                message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message:
                    if message.type == Gst.MessageType.ERROR:
                        error, debug = message.parse_error()
                        raise RuntimeError(f'{error}: {debug}')
                    break
        finally:
            self.pipeline.set_state(Gst.State.NULL)
