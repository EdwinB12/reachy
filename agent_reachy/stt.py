"""Transcribes speech from Reachy Mini's microphone using faster-whisper.

Configured via the `stt:` section of voice_config.yaml at the project root
(see voice_config.example.yaml for the template). Utterances are segmented
using the ReSpeaker mic array's hardware voice-activity detection
(ReachyMini.media.get_DoA()) when available, falling back to a simple
amplitude threshold otherwise.
"""

import time
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from reachy_mini import ReachyMini

CONFIG_PATH = Path(__file__).resolve().parent.parent / "voice_config.yaml"

DEFAULT_CONFIG = {
    "enabled": False,
    "model_size": "base.en",
    "device": "cpu",
    "compute_type": "int8",
    "silence_duration_s": 0.8,
    "min_speech_duration_s": 0.3,
    "amplitude_threshold": 0.02,
}

_model = None
_model_key = None


def _load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH) as f:
            config.update((yaml.safe_load(f) or {}).get("stt") or {})
    except FileNotFoundError:
        pass
    return config


def _get_model(model_size: str, device: str, compute_type: str):
    global _model, _model_key
    key = (model_size, device, compute_type)

    if _model is None or _model_key != key:
        from faster_whisper import WhisperModel

        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _model_key = key

    return _model


def _to_mono(chunk: np.ndarray) -> np.ndarray:
    if chunk.ndim == 2:
        return chunk.mean(axis=1)
    return chunk


def _rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk))))


def _transcribe(model, audio: np.ndarray) -> str:
    segments, _info = model.transcribe(audio, language="en", beam_size=1)
    return " ".join(segment.text.strip() for segment in segments).strip()


def listen_loop(
    mini: ReachyMini,
    on_transcript: Callable[[str], None],
    stop_event,
    speaking_event,
) -> None:
    """Continuously listen to Reachy Mini's microphone and report transcripts.

    Runs until `stop_event` is set. Calls `on_transcript(text)` for each
    completed utterance. Skips capturing audio while `speaking_event` is
    set, so the robot doesn't transcribe its own voice.

    Does nothing (returns immediately) if STT is disabled (missing/absent
    voice_config.yaml, missing `stt:` section, or "enabled": false in it),
    matching tts.speak()'s disabled behavior. Unlike TTS, this config is
    only read once at startup, since the underlying recording pipeline
    isn't cheap to start/stop on every check.
    """
    config = _load_config()
    if not config["enabled"]:
        return

    model = _get_model(config["model_size"], config["device"], config["compute_type"])
    silence_duration_s = config["silence_duration_s"]
    amplitude_threshold = config["amplitude_threshold"]

    mini.media.start_recording()
    try:
        samplerate = mini.media.get_input_audio_samplerate()
        min_speech_samples = config["min_speech_duration_s"] * samplerate

        # Recording runs continuously in the background, so discard any
        # backlog before we start segmenting utterances.
        flush_until = time.monotonic() + 0.2
        while time.monotonic() < flush_until:
            mini.media.get_audio_sample()

        speech_chunks = []
        in_speech = False
        silence_start = None
        doa_available = True

        while not stop_event.is_set():
            if speaking_event.is_set():
                speech_chunks = []
                in_speech = False
                silence_start = None
                time.sleep(0.05)
                continue

            chunk = mini.media.get_audio_sample()
            if chunk is None:
                time.sleep(0.01)
                continue

            mono_chunk = _to_mono(chunk)

            speech_now = None
            if doa_available:
                doa = mini.media.get_DoA()
                if doa is None:
                    doa_available = False
                else:
                    _, speech_now = doa
            if speech_now is None:
                speech_now = _rms(mono_chunk) > amplitude_threshold

            if speech_now:
                speech_chunks.append(mono_chunk)
                in_speech = True
                silence_start = None
            elif in_speech:
                speech_chunks.append(mono_chunk)
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= silence_duration_s:
                    utterance = np.concatenate(speech_chunks)
                    speech_chunks = []
                    in_speech = False
                    silence_start = None

                    if len(utterance) >= min_speech_samples:
                        text = _transcribe(model, utterance)
                        if text:
                            on_transcript(text)
    finally:
        mini.media.stop_recording()
