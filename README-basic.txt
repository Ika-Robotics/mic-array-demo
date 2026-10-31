UMA-16 v2 + camera: local sound-direction demo

Double-click Start Demo.cmd to open the live window. Close it with Esc.
The array must be plugged in after the miniDSP driver is installed.
The app requires the 16-channel miniDSP ASIO input and the external
"icspring camera". It will not silently use your laptop microphone/webcam.

Try speaking, clapping, or rustling paper 1-2 metres in front of the array,
then move the sound to the left and right. A green crosshair marks the
strongest estimated direction; an orange overlay shows relative acoustic
response. Each of the 16 bars shows an actual microphone level in dBFS.
This is a basic direction demonstration, not a calibrated acoustic camera.

If the camera says Access denied, close the demo and start it yourself
from File Explorer. The coding session has restricted device access.
If it still fails, check Windows Settings > Privacy & security > Camera:
camera access and access for desktop apps must be enabled. Close other
applications using this camera. Do not disable other security protections.

Controls (click the demo window first):
  [ / ]   lower/raise the detection gate by 3 dB (lower = more sensitive)
  - / +   adjust assumed camera horizontal field of view by 5 degrees
  R       rotate the direction mapping 90 degrees
  X / Y   flip horizontal/vertical direction mapping
  H       toggle acoustic heatmap
  Esc     stop capture and close

Alignment: make a sound at the left, right, top and bottom of the camera
view. Use R, X and Y if the board is rotated relative to the camera.
The initial camera field of view is an uncalibrated 80 degrees. Adjust
it with - / +. Settings currently reset when you close the application.

Limitations: the flat array cannot distinguish front from back, estimate
distance, or reliably separate multiple simultaneous sources. Reflections,
fans and other background sounds can dominate. Use one clear source in
front of the board, approximately 1 metre away or farther. The crosshair
is suppressed for quiet, diffuse, stale or clipped input. "Outside camera
view" means the strongest scanned direction is outside the assumed FOV.
The map is a relative correlation response, not calibrated sound pressure.
Audio and video are live but not hardware synchronized.

Implementation: native Windows C++17 / Media Foundation / GDI, with the
existing PortAudio ASIO DLL under ../ika-simulator/.venv/Lib/site-packages/.
No Python packages or audio drivers were changed. Raw audio/video stays
in memory during the live demo; there is no network streaming or recording.
The optional diagnostic --save-preview writes a single local UI image.

Processing: 16 synchronous channels at 48 kHz, 8192-sample blocks, four
2048-point FFT windows, pairwise GCC-PHAT and a steered-response scan over
51 x 35 forward rays, using 700-3200 Hz and 42 mm microphone spacing.
The channel map follows the UMA-16 v2 microphone-side drawing:
  8  7 10  9
  6  5 12 11
  4  3 14 13
  2  1 16 15
This drawing is mirrored horizontally for a camera looking outward.

Build: run build.cmd (uses the installed Visual Studio 2022 Build Tools).
Tests: MicArrayDemo.exe --self-test
Hardware check: MicArrayDemo.exe --check
The hardware check exits 2 if the camera is unavailable, even if audio works.

References:
https://www.minidsp.com/images/documents/UMA-16%20v2%20User%20Manual.pdf
https://learn.microsoft.com/en-us/windows/win32/medfound/enumerating-video-capture-devices
https://learn.microsoft.com/en-us/windows/win32/medfound/processing-media-data-with-the-source-reader
