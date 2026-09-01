# Reachy Mini

This robot is the wireless version, reached over Ethernet at `reachy-mini.local`
(its WiFi stays in unused hotspot mode — see [reachy_mini_fix.py](reachy_mini_fix.py)
for why that matters for media/WebRTC).

## Wake up / shut down

The daemon on the robot must be running (`state: "running"`) before the SDK
(`ReachyMini()`) can connect. Check its current state:

```bash
curl http://reachy-mini.local:8000/api/daemon/status
```

**Wake up** (starts the control backend and moves the robot to its awake pose):

```bash
curl -X POST "http://reachy-mini.local:8000/api/daemon/start?wake_up=true"
```

**Shut down** (moves the robot to its sleep pose, then stops the control backend):

```bash
curl -X POST "http://reachy-mini.local:8000/api/daemon/stop?goto_sleep=true"
```

Pass `wake_up=false` / `goto_sleep=false` instead if you want to start/stop the
backend without the wake/sleep move.

# Start Agent

```bash
python3 -m agent_reachy
```

## Voice output

By default the agent only prints its replies to the terminal. To have
Reachy Mini speak its replies out loud (via [Piper](https://github.com/OHF-Voice/piper1-gpl)),
copy the example config and enable it:

```bash
cp voice_config.example.yaml voice_config.yaml
```

Then under the `tts:` section of `voice_config.yaml` set `enabled: true`, and
download a voice model into `voices/` (matching `model_path` in the config,
default `voices/en_US-lessac-medium.onnx`):

```bash
python3 -m piper.download_voices en_US-lessac-medium --download-dir voices/
```

Browse [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
for other voices/languages. Set `enabled: false` under `tts:` in
`voice_config.yaml` (or delete the file) to go back to text-only output — no
restart required, it's re-read on every reply.

## Voice input

By default the agent only accepts typed commands. To also let you talk to
Reachy Mini (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
running locally on CPU), copy the example config (if you haven't already)
and enable it:

```bash
cp voice_config.example.yaml voice_config.yaml
```

Then under the `stt:` section of `voice_config.yaml` set `enabled: true` and
restart the agent (unlike TTS, this config is only read once at startup).
Once running, just speak near the robot — it segments your speech using the
ReSpeaker mic array's hardware voice detection (falling back to a simple
volume threshold if that hardware isn't available), transcribes it, and
feeds it into the same command pipeline as typed input. Typed input keeps
working at the same time.

The robot ignores the microphone while it's speaking its own replies, so it
doesn't transcribe itself. Note that voice input and the
`record_and_playback_audio` tool (the mic/speaker self-test) can't be used
at the same time — both need exclusive control of the recording pipeline.

## Troubleshooting

If `ReachyMini()` raises a `ConnectionError`/`TimeoutError`, check
`/api/daemon/status` first — a `"state": "stopped"` daemon is the most common
cause, fixed by the wake-up command above.
