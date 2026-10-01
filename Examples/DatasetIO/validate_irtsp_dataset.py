#!/usr/bin/env python3
"""Read back every PNG and audit source clocks without changing the dataset."""
import argparse
import csv
import json
from pathlib import Path
import struct
import zlib


def validate(path):
    path = Path(path)
    result = {}
    failures = []
    rows = {}
    for kind, nominal in [('camera',30), ('gyro',100), ('accel',100), ('intrinsics',30)]:
        with (path / (kind+'.csv')).open() as f:
            rows[kind] = list(csv.DictReader(f))
        times = [float(r['host_ts']) for r in rows[kind]]
        dt = [b-a for a,b in zip(times,times[1:])]
        result[kind] = dict(count=len(times), rate_hz=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
            regressions=sum(d<0 for d in dt), duplicates=sum(d==0 for d in dt),
            max_gap_s=max(dt,default=None), gaps_over_1_5_periods=sum(d>1.5/nominal for d in dt))
        rate = result[kind]['rate_hz']
        if len(times)<2 or rate is None or not .8*nominal <= rate <= 1.2*nominal or any(d<=0 for d in dt) or any(d>1.5/nominal for d in dt):
            failures.append(kind+' timestamp/count/gap check')
    intrinsic_times = {float(r['host_ts']) for r in rows['intrinsics']}
    dimensions = set()
    for row in rows['camera']:
        data = (path / row['filename']).read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        offset, compressed, size = 8, bytearray(), None
        while offset < len(data):
            length = struct.unpack('>I',data[offset:offset+4])[0]
            kind = data[offset+4:offset+8]
            payload = data[offset+8:offset+8+length]
            crc = struct.unpack('>I',data[offset+8+length:offset+12+length])[0]
            assert zlib.crc32(kind+payload)&0xffffffff == crc
            if kind == b'IHDR':
                w,h,depth,color,_,_,interlace = struct.unpack('>IIBBBBB',payload)
                assert (depth,color,interlace)==(8,2,0)
                size = (w,h)
            if kind == b'IDAT': compressed.extend(payload)
            offset += length+12
        pixels = zlib.decompress(compressed)
        assert len(pixels)==size[1]*(1+size[0]*3)
        assert size == (int(row['width']),int(row['height']))
        dimensions.add(size)
        assert float(row['intrinsics_host_ts']) in intrinsic_times
        assert abs(float(row['intrinsics_host_ts'])-float(row['host_ts'])) <= 12e-6
    result['decoded_png_count'] = len(rows['camera'])
    result['dimensions'] = sorted(dimensions)
    result['max_intrinsics_delta_us'] = max((abs(float(r['intrinsics_delta_s']))*1e6 for r in rows['camera']),default=None)
    info = json.loads((path/'session_info.json').read_text())
    if not info['complete'] or info['source_sequence_gaps'] or info['dropped_writer_frames'] or any(info['dropped_acquisition_events'].values()):
        failures.append('session integrity/drop checks')
    for kind in rows:
        if len(rows[kind]) != info[kind+'_count']:
            failures.append(kind+' metadata count')
    result['failures'] = failures
    print(json.dumps(result,indent=2))
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    raise SystemExit(0 if validate(parser.parse_args().dataset) else 1)
