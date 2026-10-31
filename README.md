# mic-array-demo

Hackathon experiments for Ika: a miniDSP **UMA-16 v2** 16-microphone array plus a USB
camera. The aim is to tell apart up to three people in front of the robot, focus on
each person's voice (beamforming steered by their face), and caption what they say.

| Folder | What |
|---|---|
| `vision/` | Python: face/lip/head-pose tracking, face-steered beamforming (DAS/MVDR/LCMV + noise filter), quick acoustic calibration with clicks/whistles, "ME" re-identification, record and A/B playback (general mix vs focused), Gemini captions. See [vision/README.md](vision/README.md). |
| root | Native C++ (Win32 / Media Foundation / PortAudio ASIO) tools: setup wizard, sound-direction demo and focus app. See `README.txt`, `SPEECH-FOCUS.txt`, `IKA-INTEGRATION.txt`. |

## Python quick start (Windows)

```
cd vision
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe fetch_assets.py
.venv\Scripts\python.exe app_selftest.py
```

Then run `Start Beam Demo.cmd` with the array, camera and headphones connected.
ASIO input uses the PortAudio ASIO build bundled with `sounddevice`.

## Captions / API key

Captions read `GOOGLE_API_KEY` from the environment or from
`%LOCALAPPDATA%\Ika\simulator\.env` (the same file ika-simulator uses). No key is
stored in this repository; never commit one.

## Not in the repo (on purpose)

Voice recordings, face templates (`me.npz`), local calibration files, build outputs,
generated screenshots, and the third-party models and test images (downloaded by
`vision/fetch_assets.py`).
