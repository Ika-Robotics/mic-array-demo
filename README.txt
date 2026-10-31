IKA MICROPHONE + CAMERA SETUP WIZARD

NEW DEFAULT: FIXED MOUTH / SPEECH FOCUS
Start Demo.cmd and Start Speech Focus.cmd now open a direct geometry-based
view with mouth selection and ten-second original/focused audio comparisons.
No room calibration is required. See SPEECH-FOCUS.txt for the current workflow.
build.cmd now produces MicArrayFocus-Buffered.exe; --wizard retains the setup below.
Recordings are created only when requested; saved wizard profiles are preserved.

LEGACY SETUP WIZARD
Reconnect the UMA-16 v2 and external icspring camera. Close the basic demo and
other applications using the devices, then double-click Start Setup Wizard.cmd.
Run it from File Explorer: this coding session cannot activate the camera.

DISPLAY SIZE
Text, buttons and status bars scale together with the window. Maximize for a
larger interface. Windows display scaling is used for the initial window size;
moving between monitors updates the layout without stretching a screenshot.
The launchers now use MicArrayFocus-Buffered.exe. Close the previous wizard
before reopening to apply the update. Unsaved setup progress must be repeated;
previously saved profiles remain in setup-profile.ini.

1. Devices
Wait for live external video and all 16 microphone bars. No laptop microphone
or webcam fallback is used. Retry failed devices after reconnecting if needed.

2. Room noise
Keep normal fans and room noise running. Click Measure room; stay quiet through
the three-second countdown and six-second measurement. Unstable/clipped input
is rejected. The gate is set 6 dB above the upper measured background level.
This is a sound-detection threshold, not speech classification.

3. Camera alignment -- phone tone mode (default)
Play a continuous 440 Hz SINE tone on your phone. Use a tone-generator app or
copy 440Hz-calibration.wav to the phone (60 seconds; replay/loop as needed).
Keep the phone silent during the room-noise measurement, then start playback.
Hold it at least 1 metre in front of the array, with its speaker facing the
array. Start at a moderate volume and increase slightly if there is no lock.
Move the phone LEFT, RIGHT, HIGH, then LOW in the camera image. For each:
  - Wait for "440 Hz: LOCKED" in the status panel.
  - Click the phone SPEAKER's actual position, not the suggested marker.
  - Click Capture position.
  - After the three-second countdown, keep the sound steady for four seconds.
The four observations fit image alignment, rotation/reflection, scale and offset.
Poor or unsteady captures must be repeated. Keep playback continuous throughout
the countdown and capture. Tone mode uses a separate frequency/phase detector;
it does not require the tone to exceed the broadband room-noise gate.
"Tone clarity" compares the narrow tone with nearby frequency bins, not total
room volume. A lock requires a near-440 Hz peak, at least 12 clear microphone
channels, consistent spatial phases and an unclipped input.
It cannot distinguish the phone from another source at the same frequency.
Reflections can still bias a locked direction; independent verification remains
mandatory. This mode is intended to help in crowds, not guarantee calibration.

The Source button switches back to the original broadband mode (steady speech
or rustling paper); changing modes clears alignment points and verification.
Broadband mode uses 700-3200 Hz and does not locate a pure 440 Hz tone.

4. Independent verification
Keep the tone playing. Move the phone to a NEW point near the center, wait for
lock, click the speaker and capture again. In broadband mode use the same
broadband source as during alignment.
This fifth point is not used to fit alignment. Excessive error blocks completion.
Go back to redo alignment if a repeated check still fails.

5. Interaction area
STOP the phone tone. Drag a rectangle where a person will interact with Ika, usually around their
expected face/upper-body position. Minimum size is 12% of each image dimension.
This is a geometric region, not a person detector.

6. Location tests
Click Test inside and speak continuously for eight seconds after countdown.
Then Test outside from outside the rectangle, still in front of the array.
These tests and the live view use ordinary broadband sound, not the phone-tone
filter. Crowd noise can still affect them. They validate location, not whether
a sound is speech. Clear detections
must land inside at least 60% during the inside trial and no more than 35%
during the outside trial. Silence/uncertain blocks are excluded from those
ratios; a minimum number and fraction of usable observations is also required.

7. Review and save
Review room level, held-out alignment error and trial results, then Save.
setup-profile.ini is written atomically. Despite the filename, this is a private
versioned text format, not general INI syntax; do not edit it manually. The
previous saved profile remains intact until Save.

SUBSEQUENT RUNS
Start Demo.cmd: opens the new mouth-focus view; see SPEECH-FOCUS.txt.
Start Setup Wizard.cmd: reruns setup, keeping the saved file until Save.
Start Basic Demo.cmd: opens the original demo (close the wizard first).

The profile is bound to the array USB instance, camera symbolic link and video
resolution. A mismatch requires setup again. Rerun if the camera moves relative
to the array, the mounting changes, or the room background changes.

The live view has a relative acoustic heatmap, smoothed marker, interaction
rectangle and 16 level bars. The marker box is a visual indication based on
setup error, not a statistical confidence interval. The map is relative
correlation response, not calibrated sound pressure.
More/less sensitive changes the current gate. Retest and save requires new
inside/outside trials before persisting it. Esc closes the app.

LIMITATIONS AND IKA
This version locates sound. It does not classify voice, detect/identify people,
transcribe speech, or recognize who is talking. The flat array has front/back
ambiguity. Alignment is empirical, not full camera/lens calibration.
Reflections, multiple/nearby sources and lens distortion can reduce accuracy.
Audio and video are not hardware synchronized.
The future Ika decision needs speech detection, direction and visual person
association. Exported speech/person-speaking fields deliberately remain unknown.
See IKA-INTEGRATION.txt. No Ika service is connected or modified yet.

PRIVACY AND DIAGNOSTICS
Raw audio/video stays in memory. Setup saves numeric summaries, device IDs,
geometry and settings, not recordings. No network service is started.
ika-audio-status.json is the new status file; live-status.txt is from the old demo.

Build: build.cmd (installed Visual Studio 2022 Build Tools and Windows SDK)
Tests: MicArrayFocus-Buffered.exe --setup-tests
       MicArrayFocus-Buffered.exe --self-test
       MicArrayFocus-Buffered.exe --tone-tests
UI QA: MicArrayFocus-Buffered.exe --render-wizard-tests
Tone file: MicArrayFocus-Buffered.exe --make-phone-tone
Inventory: MicArrayFocus-Buffered.exe --array-id
Hardware diagnostic: MicArrayFocus-Buffered.exe --check (close other demos first)
Test artifacts are under test-output; generated grids are not camera captures.

Runtime uses the existing PortAudio ASIO DLL under:
../ika-simulator/.venv/Lib/site-packages/_sounddevice_data/portaudio-binaries/
No driver changes or package downloads were needed for the wizard.

Sources:
https://www.minidsp.com/images/documents/UMA-16%20v2%20User%20Manual.pdf
https://learn.microsoft.com/en-us/windows/win32/medfound/enumerating-video-capture-devices
