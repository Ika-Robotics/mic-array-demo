"""Read-only device inventory and short, in-memory 16-channel capture via PortAudio."""
import ctypes as c
import json
import math
from pathlib import Path

DLL = Path(__file__).resolve().parents[1] / 'ika-simulator/.venv/Lib/site-packages/_sounddevice_data/portaudio-binaries/libportaudio64bit-asio.dll'

class Device(c.Structure):
    _fields_ = [('version', c.c_int), ('name', c.c_char_p), ('host', c.c_int),
                ('inputs', c.c_int), ('outputs', c.c_int), ('low_in', c.c_double),
                ('low_out', c.c_double), ('high_in', c.c_double),
                ('high_out', c.c_double), ('rate', c.c_double)]

class Host(c.Structure):
    _fields_ = [('version', c.c_int), ('type', c.c_int), ('name', c.c_char_p),
                ('count', c.c_int), ('input', c.c_int), ('output', c.c_int)]

class Params(c.Structure):
    _fields_ = [('device', c.c_int), ('channels', c.c_int), ('format', c.c_ulong),
                ('latency', c.c_double), ('info', c.c_void_p)]

def main():
    p = c.CDLL(str(DLL))
    p.Pa_GetDeviceInfo.restype = c.POINTER(Device)
    p.Pa_GetHostApiInfo.restype = c.POINTER(Host)
    p.Pa_GetErrorText.restype = c.c_char_p
    p.Pa_OpenStream.argtypes = [c.POINTER(c.c_void_p), c.POINTER(Params), c.c_void_p,
                               c.c_double, c.c_ulong, c.c_ulong, c.c_void_p, c.c_void_p]
    p.Pa_StartStream.argtypes = [c.c_void_p]
    p.Pa_ReadStream.argtypes = [c.c_void_p, c.c_void_p, c.c_ulong]
    p.Pa_StopStream.argtypes = [c.c_void_p]
    p.Pa_CloseStream.argtypes = [c.c_void_p]
    def check(code):
        if code < 0:
            raise RuntimeError(p.Pa_GetErrorText(code).decode())
    check(p.Pa_Initialize())
    stream = c.c_void_p()
    try:
        candidates = []
        for i in range(p.Pa_GetDeviceCount()):
            d = p.Pa_GetDeviceInfo(i).contents
            host = p.Pa_GetHostApiInfo(d.host).contents.name.decode()
            name = d.name.decode(errors='replace')
            if any(s in name.lower() for s in ['minidsp', 'mchstream', 'uma16']):
                print(json.dumps({'index': i, 'name': name, 'host': host, 'inputs': d.inputs}), flush=True)
                if d.inputs >= 16:
                    candidates.append((host != 'ASIO', i))
        if not candidates:
            raise RuntimeError('No miniDSP input with 16 channels found')
        _, index = min(candidates)
        d = p.Pa_GetDeviceInfo(index).contents
        params = Params(index, 16, 1, d.high_in, None)
        check(p.Pa_OpenStream(c.byref(stream), c.byref(params), None, 48000., 1024, 0, None, None))
        check(p.Pa_StartStream(stream))
        data = (c.c_float * (1024 * 16))()
        sums = [0.] * 16
        peaks = [0.] * 16
        overflow = 0
        for _ in range(94):
            rc = p.Pa_ReadStream(stream, data, 1024)
            if rc == -9981:
                overflow += 1
            else:
                check(rc)
            for k, value in enumerate(data):
                ch = k % 16
                sums[ch] += value * value
                peaks[ch] = max(peaks[ch], abs(value))
        report = {'device_index': index, 'sample_rate': 48000, 'channels': 16,
                  'seconds': 94*1024/48000, 'overflows': overflow,
                  'rms_dbfs': [round(10*math.log10(max(s/(94*1024), 1e-20)), 1) for s in sums],
                  'peaks': peaks}
        print(json.dumps(report, indent=2), flush=True)
        Path(__file__).with_name('audio-check.json').write_text(json.dumps(report, indent=2))
    finally:
        if stream.value:
            p.Pa_StopStream(stream)
            p.Pa_CloseStream(stream)
        p.Pa_Terminate()

if __name__ == '__main__':
    main()
