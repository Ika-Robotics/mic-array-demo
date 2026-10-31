# Vision: faces, lips, head pose, direction

Self-contained (own `.venv`, Python 3.12, mediapipe 1.2 + OpenCV 5). Does not touch
the C++ wizard or the root `.venv`.

## Run
- Live: double-click `Start Vision Demo.cmd` (from File Explorer — the coding session
  cannot activate the camera). Close the wizard/basic demo first if they hold the camera.
- Offline: `.venv\Scripts\python.exe vision_demo.py --input some.jpg|some.mp4`
- Tests: `.venv\Scripts\python.exe selftest.py` (18 checks, no camera)

Keys: `[` `]` lip threshold, `-` `+` camera FOV, `l` lip outline, `m` mirror, `Esc` quit.

## Pipeline (per frame, ~30-45 ms for 3 faces on the Ryzen AI 9 HX 370)
1. **YuNet** face detector on the full frame (every 4th frame or when a face is lost).
   Finds faces down to ~25 px wide (~3+ m). MediaPipe's own detector misses faces
   under ~130 px at 1280x720, i.e. anyone beyond ~1 m with this camera.
2. **MediaPipe FaceLandmarker** on a 256 px crop around each face: 478 landmarks,
   blendshapes (jawOpen), head transform.
3. **Lip activity** = std-dev of inner-lip gap / eye span over 0.6 s. Talking oscillates
   (3-6 Hz); a held-open mouth, smile or yawn does not. Hysteresis + 0.35 s hold.
4. **Head pose / looking at camera**: `off_axis_deg` = angle between the head's forward
   axis and the line from the face to the camera, so someone at the frame edge facing
   straight ahead is correctly *not* looking at the camera. Uses `--hfov`.
5. **Direction**: the *mouth* position is mapped to an array ray using the wizard's
   `../setup-profile.ini` affine alignment when saved; otherwise a pinhole with `--hfov`.
   `unit_vector` is in the wizard's array frame and can steer a beamformer directly.
6. **Range**: from outer eye-corner span (~9 cm) and `--hfov`, roughly ±20%.

`addressing_camera_visual = looking_at_camera AND mouth_moving`. Combine it with audio
speech detection on that face's beam for "speaking to Ika".

## Output: `ika-vision-status.json` (~10 Hz, atomic replace)
`timestamp_ms`, `monotonic_ms`, `camera_fresh`, `fps`, `direction_source`, and `faces[]`
(left to right) with `id`, `bbox_norm`, `mouth_norm`, `lip_activity`, `mouth_moving`,
`head_pose{yaw,pitch,roll,off_axis}`, `looking_at_camera`, `addressing_camera_visual`,
`distance_m_est`, `array_direction{unit_vector, azimuth_deg, elevation_deg, source}`.
For the audio pipeline, importing `face_tracker.FaceTracker` in-process is better than
the file (lower latency, frame timestamps).

## To tune on the real camera
- `--hfov`: measure the icspring lens FOV. Default 80 is a guess; it affects
  looking-at-camera for off-centre faces, the fallback direction, and range.
- Lip thresholds (`[` `]`): watch the bar under each face. White ticks = off/on thresholds.
  Check silent people sitting still stay left of the first tick.
- Look thresholds (`Settings.look_on_deg/look_off_deg`, 20/28 deg).

## Known limits
- Verified on still/synthetic images only, not yet on live video of real talkers.
- Head pose, not eye gaze. Someone looking at the camera with eyes only (head turned)
  reads as "away".
- Camera and audio are not hardware-synchronised. Use the timestamps and allow ~50-100 ms.
- Only one process can hold the camera. Long term, the vision and audio pipelines
  should run in one app.

---

# Focus demo: find me, beam at my mouth, compare with the general mix

`Start Beam Demo.cmd`. Before starting:
- Use **earbuds/headphones**. Laptop speakers would feed back into the array.
- Close every other app using the camera or the miniDSP ASIO driver (wizard,
  MicArrayFocus, vision demo).
- Plug in the charger. On battery the face tracker runs at about half speed.

Press `h` in the app for the key list.

## Suggested flow
1. **Click your face**, press **`e`**: you are enrolled as ME (SFace face recognition,
   saved in `me.npz`). From then on ME is found and followed automatically, even after
   leaving and coming back or swapping seats. `E` forgets ME.
2. **Calibrate (`c`)**: click your tongue, snap or whistle near your mouth while moving
   your head left/right (and a little up/down). Each sound gives a green dot at your mouth;
   the orange line shows the fit error. When the text turns green, press `c` to save
   (`beam-calibration.json`, takes priority over the wizard's `setup-profile.ini`, which is
   never modified). `Esc` cancels. Re-do it if the camera or array moves.
3. **Listen**: `space` switches **general** (plain mean of all 16 mics) and **focused**
   (beam on ME + noise filter); `a` alternates every 3 s. `b` changes beam type, `n`
   toggles the noise filter.
4. **Record & compare**: `r` starts, `r` stops. About 2 s later playback is ready: `p` plays,
   `space` switches general/focused at the same position. Files are also in
   `recordings/<time>/two/{general,focused}.wav`, with voice loudness matched.
5. **Captions (`t`)**: Gemini transcribes each face's beam, gated by that face's lip
   movement. Other people's speech leaking into your beam is not captioned while your
   lips are still.

## Captions (captioner.py)
- Key: `GOOGLE_API_KEY` from `%LOCALAPPDATA%\Ika\simulator\.env` (same as ika-simulator).
- Model `gemini-3.8-flash` (`--caption-model`). It falls back to 3.7-flash, 3.5-flash, then
  3.5-transcribe on 503/429/timeout, and skips a busy model for 60 s.
- On 2026-10-30 the key got **429 quota exceeded** on 3.8 and 3.7; 3.5-flash answered in 2-14 s.
- Captions use the beam **without** the noise filter. On a real recording the filtered audio
  lost words in transcription.
- Offline: `python caption_recording.py recordings/<time>` (general vs focused transcripts).

## Files
`beamformer.py` (das/mvdr/lcmv, PostFilter, mix), `calibration.py`, `identity.py`,
`captioner.py`, `pair_render.py` (general/focused from a recording), `make_two.py`,
`render_recording.py` (every person x beam type).

Tests (no hardware needed): `selftest.py` (vision), `beam_selftest.py`,
`calibration_selftest.py`, `caption_selftest.py`, `app_selftest.py` (whole app with fake
camera/audio: enroll, swap seats, mirrored-mount calibration, record, playback, captions).
