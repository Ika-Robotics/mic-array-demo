"""Offline captioner checks with a fake transport (no network): python caption_selftest.py"""
import threading
import time

import numpy as np

import captioner as cm
from captioner import Captioner, TransportError, clean_text

HOP, FS = 512, 48000
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")


class Fake:
    """Records calls; behaviour per model: 'ok', '503', 'timeout', '404'."""

    def __init__(self, behaviour=None, text="hello there"):
        self.behaviour = behaviour or {}
        self.calls = []
        self.audio_s = []
        self.text = text
        self.lock = threading.Lock()

    def __call__(self, model, wav, prompt, timeout):
        with self.lock:
            self.calls.append(model)
            self.audio_s.append((len(wav) - 44) / 2 / 16000)
        b = self.behaviour.get(model, "ok")
        if b == "503":
            raise TransportError(503, "HTTP 503 high demand")
        if b == "timeout":
            raise TransportError(None, "timeout")
        if b == "404":
            raise TransportError(404, "HTTP 404")
        if b == "204":
            raise TransportError(204, "no content")
        return self.text


def run(cap, pattern, face=1, t0=100.0):
    """pattern: list of (seconds, speaking). Feeds 1 kHz tone blocks."""
    t = t0
    tone = (0.01 * np.sin(2 * np.pi * 1000 * np.arange(HOP) / FS)).astype(np.float32)
    for secs, speaking in pattern:
        for _ in range(int(round(secs * FS / HOP))):
            cap.feed(face, tone, speaking, t)
            t += HOP / FS
    return t


def wait(cap, timeout=5):
    end = time.time() + timeout
    while time.time() < end and any(c.pending for c in cap.all_captions()):
        time.sleep(0.01)


# 1. Pre-roll, hangover close, timing
fake = Fake()
cap = Captioner(transport=fake)
run(cap, [(1.0, False), (2.0, True), (0.5, False)])
check("utterance still open during 0.5 s pause (< 0.7 s hangover)", not cap.all_captions())
run(cap, [(0.3, False)], t0=103.5)
wait(cap)
caps = cap.all_captions()
check("utterance closes after 0.7 s of silence", len(caps) == 1 and not caps[0].pending, f"({len(caps)})")
if caps:
    c = caps[0]
    check("t_start includes ~0.4 s pre-roll", abs(c.t_start - (101.0 - 0.4)) < 0.02, f"(t_start {c.t_start:.3f})")
    check("t_end at end of speech", abs(c.t_end - 103.0) < 0.03, f"(t_end {c.t_end:.3f})")
    check("audio sent = pre-roll + speech + ~0.25 s tail", abs(fake.audio_s[0] - 2.65) < 0.05, f"({fake.audio_s[0]:.2f} s)")
    check("text delivered with model name", c.text == "hello there" and c.model == "gemini-3.8-flash")
cap.close()

# 2. Short blips are dropped
fake = Fake()
cap = Captioner(transport=fake)
run(cap, [(0.5, False), (0.25, True), (1.0, False)])
wait(cap)
check("0.25 s blip dropped (< 0.4 s speech)", not cap.all_captions() and not fake.calls)
cap.close()

# 3. Force split at 10 s, continuous speech keeps going
fake = Fake()
cap = Captioner(transport=fake)
run(cap, [(23.0, True), (1.0, False)])
wait(cap)
caps = cap.all_captions()
check("23 s monologue -> 3 utterances (10 + 10 + 3)", len(caps) == 3, f"({[round(c.t_end - c.t_start, 1) for c in caps]})")
check("split utterances are contiguous", len(caps) == 3 and abs(caps[1].t_start - caps[0].t_end) < 0.02)
cap.close()

# 4. Separate faces are segmented independently; captions(face_id) filters
fake = Fake()
cap = Captioner(transport=fake)
tone = np.zeros(HOP, np.float32)
t = 0.0
for i in range(int(3 * FS / HOP)):
    cap.feed(1, tone, i * HOP / FS < 1.5, t)
    cap.feed(2, tone, i * HOP / FS > 1.0, t)
    t += HOP / FS
cap.flush()
wait(cap)
check("two faces -> one utterance each", len(cap.captions(1)) == 1 and len(cap.captions(2)) == 1)
cap.close()

# 5. Fallback on 503 and busy-skip
fake = Fake({"gemini-3.8-flash": "503"})
cap = Captioner(transport=fake, max_workers=1)
run(cap, [(1.0, True), (1.0, False)])
wait(cap)
run(cap, [(1.0, True), (1.0, False)], t0=200)
wait(cap)
caps = cap.all_captions()
check("503 on 3.8 -> answered by 3.7", caps and caps[0].model == "gemini-3.7-flash", f"({[c.model for c in caps]})")
check("busy 3.8 skipped on next utterance", fake.calls == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.7-flash"],
      f"({fake.calls})")
check("status explains fallback", "busy" in cap.status and "3.7" in cap.status, f"({cap.status!r})")
cap.close()

# 6. All models fail -> error caption, never raises
fake = Fake({m: "timeout" for m in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-transcribe")})
cap = Captioner(transport=fake)
run(cap, [(1.0, True), (1.0, False)])
wait(cap)
caps = cap.all_captions()
check("all models down -> caption with error, not pending", len(caps) == 1 and caps[0].error and not caps[0].pending,
      f"({caps[0].error[:60] if caps else ''})")
cap.close()

# 7. Retry rounds (offline tool): recovers once the service comes back
class Flaky(Fake):
    def __call__(self, model, wav, prompt, timeout):
        if len(self.calls) < 4:
            with self.lock:
                self.calls.append(model)
            raise TransportError(503, "busy")
        return super().__call__(model, wav, prompt, timeout)
fake = Flaky()
cap = Captioner(transport=fake, retry_rounds=3, retry_backoff_s=(0.05,))
run(cap, [(1.0, True), (1.0, False)])
wait(cap)
caps = cap.all_captions()
check("retry round recovers after all models were busy", caps and caps[0].text == "hello there", f"({len(fake.calls)} calls)")
cap.close()

# 7b. "No content" answers: next model tried; if nobody has text, accept empty (not an error)
fake = Fake({"gemini-3.8-flash": "204"})
cap = Captioner(transport=fake)
run(cap, [(1.0, True), (1.0, False)])
wait(cap)
c = cap.all_captions()[0]
check("no-content from 3.8 -> 3.7 transcribes", c.text == "hello there" and c.model == "gemini-3.7-flash")
cap.close()
fake = Fake({"gemini-3.8-flash": "204", "gemini-3.7-flash": "503", "gemini-3.5-flash": "204", "gemini-3.5-transcribe": "204"})
cap = Captioner(transport=fake)
run(cap, [(1.0, True), (1.0, False)])
wait(cap)
c = cap.all_captions()[0]
check("only no-content answers -> empty caption, no error", c.text == "" and not c.error and not c.pending, f"({c.model})")
cap.close()

# 8. No key -> disabled, feed is a no-op
saved = cm.load_api_key
cm.load_api_key = lambda env_file=None: ""
cap = Captioner()
cm.load_api_key = saved
run(cap, [(1.0, True), (1.0, False)])
check("no key -> disabled with status", not cap.enabled and "no API key" in cap.status and not cap.all_captions())
cap.close()

# 9. Text cleanup
check("clean_text strips no-speech markers",
      clean_text("[no speech]") == "" and clean_text("  (inaudible) ") == "" and
      clean_text('"Hello [noise] world"') == "Hello world" and clean_text("Silence.") == "")

# 10. feed() cost on the audio thread
class Slow(Fake):
    def __call__(self, *a):
        time.sleep(0.5)
        return "x"
cap = Captioner(transport=Slow())
blk = np.zeros(HOP, np.float32)
n, t = 20000, 0.0
t0 = time.perf_counter()
for i in range(n):
    cap.feed(i % 3, blk, (i // 200) % 2 == 0, t)   # utterances opening/closing constantly
    t += HOP / FS / 3
ms = (time.perf_counter() - t0) / n * 1000
check("feed() average cost < 0.2 ms with network calls in flight", ms < 0.2, f"({ms * 1000:.1f} us)")
cap.close()

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
