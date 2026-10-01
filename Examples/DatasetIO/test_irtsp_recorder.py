"""Tests of failure handling and preservation, independent of physical hardware."""
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import queue
import irtsp
from irtsp_io.recorder import DatasetRecorder
from irtsp_io.source import IrtspSource


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = DatasetRecorder(Path(self.tmp.name)/'session')
        self.rec.intrinsics = [(10., 1, dict(host_ts=10., width=2, height=2))]

    def tearDown(self):
        if self.rec.worker.is_alive():
            self.rec.finish()
        self.tmp.cleanup()

    def frame(self):
        return dict(source_host_ts=10., frame_index=0, width=2, height=2,
                    stride=6, pixels=bytes(12), decode_delivery_pc_mono_s=20., rtp_timestamp=1, ssrc=1)

    def test_geometry_refused(self):
        f = self.frame()
        f['height'] = 1
        self.rec.pending.append(f)
        with self.assertRaisesRegex(RuntimeError, 'landscape'):
            self.rec.resolve(object())

    def test_source_time_and_drain(self):
        self.rec.pending.append(self.frame())
        self.rec.resolve(object())
        info = self.rec.finish()
        with (self.rec.output/'camera.csv').open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row['timestamp_ns'], '10000000000')
        self.assertEqual(float(row['host_ts']),10.)
        self.assertTrue((self.rec.output/row['filename']).exists())
        self.assertEqual(info['camera_count'],1)

    def test_unmapped_frame_remains_inspectable(self):
        f = self.frame()
        f['source_host_ts'] = None
        self.rec.pending.append(f)
        info = self.rec.finish(object())
        self.assertEqual(info['missing_intrinsics_frame_associations'],1)
        self.assertFalse(info['complete'])
        self.assertTrue((self.rec.output/'camera/orphan_000000.png').exists())

    def test_regressions_and_duplicates(self):
        for ts in (10.,10.,9.):
            self.rec._track('gyro',ts)
        info = self.rec.finish()
        self.assertEqual(info['timestamp_integrity']['gyro'],dict(regressions=1,duplicates=1))
        self.assertFalse(info['complete'])

    def test_acquisition_overflow_visible(self):
        source = IrtspSource('unused',capacity=1)
        source.emit('record',0,None)
        source.emit('record',1,None)
        self.assertEqual(source.drops,{'record':1})
        self.assertTrue(source.stop.is_set())

    def test_raw_timestamps_remain_independent(self):
        gyro = irtsp.RawGyro(host_ts=10., unix_ts=100., seq=1, gyro=irtsp.Vec3(1,2,3))
        accel = irtsp.RawAccel(host_ts=10.004, unix_ts=100.004, seq=2, accel=irtsp.Vec3(4,5,6))
        self.rec.consume(('record',20.,gyro),object())
        self.rec.consume(('record',20.,accel),object())
        self.rec.finish()
        with (self.rec.output/'gyro.csv').open() as f:
            g = next(csv.DictReader(f))
        with (self.rec.output/'accel.csv').open() as f:
            a = next(csv.DictReader(f))
        self.assertEqual(float(g['host_ts']),10.)
        self.assertEqual(float(a['host_ts']),10.004)
        self.assertNotEqual(g['timestamp_ns'],a['timestamp_ns'])
        self.assertFalse((self.rec.output/'imu.csv').exists())

    def test_writer_overflow_visible(self):
        self.rec.pending.append(self.frame())
        with patch.object(self.rec.queue,'put_nowait',side_effect=queue.Full):
            with self.assertRaisesRegex(RuntimeError,'queue full'):
                self.rec.resolve(object())
        self.assertEqual(self.rec.finish()['dropped_writer_frames'],1)

    def test_ctrl_c_finalizes_metadata(self):
        import irtsp_recorder
        output = Path(self.tmp.name)/'interrupt'
        fake = Mock()
        fake.open.side_effect = KeyboardInterrupt
        fake.error = None
        fake.events.empty.return_value = True
        fake.drops = {}
        with patch.object(irtsp_recorder,'IrtspSource',return_value=fake), patch('sys.argv',
                ['irtsp_recorder','--host','unused','--output',str(output)]):
            self.assertEqual(irtsp_recorder.main(),1)
        fake.close.assert_called_once()
        import json
        self.assertIn('end_utc',json.loads((output/'session_info.json').read_text()))

    def test_refuse_overwrite(self):
        with self.assertRaises(FileExistsError):
            DatasetRecorder(self.rec.output)


if __name__ == '__main__':
    unittest.main()
