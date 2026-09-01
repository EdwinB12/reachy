"""Speaks the agent's replies aloud through Reachy Mini's speaker using Piper.

Configured via the `tts:` section of voice_config.yaml at the project root
(see voice_config.example.yaml for the template). All Piper-specific setup
and the enabled/disabled switch live here; callers just call speak().
"""

import tempfile
import time
import wave
from pathlib import Path

import yaml
from reachy_mini import ReachyMini

CONFIG_PATH = Path(__file__).resolve().parent.parent / "voice_config.yaml"

DEFAULT_CONFIG = {
    "enabled": False,
    "model_path": "voices/en_US-lessac-medium.onnx",
    "length_scale": 1.0,
}

_voice = None
_voice_model_path = None


def _load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH) as f:
            config.update((yaml.safe_load(f) or {}).get("tts") or {})
    except FileNotFoundError:
        pass
    return config


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = CONFIG_PATH.parent / path
    return path


def _get_voice(model_path: Path):
    global _voice, _voice_model_path

    if _voice is None or _voice_model_path != model_path:
        from piper import PiperVoice

        _voice = PiperVoice.load(str(model_path))
        _voice_model_path = model_path

    return _voice


def speak(mini: ReachyMini, text: str) -> None:
    """Synthesize `text` with Piper and play it through Reachy Mini's speaker.

    Does nothing if TTS is disabled (missing/absent voice_config.yaml,
    missing `tts:` section, or "enabled": false in it) or if `text` is
    blank.
    """
    config = _load_config()
    if not config["enabled"] or not text.strip():
        return

    from piper import SynthesisConfig

    model_path = _resolve_model_path(config["model_path"])
    voice = _get_voice(model_path)
    syn_config = SynthesisConfig(length_scale=config["length_scale"])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with wave.open(str(tmp_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)

        with wave.open(str(tmp_path), "rb") as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()

        mini.media.play_sound(str(tmp_path))
        time.sleep(duration)
    finally:
        tmp_path.unlink(missing_ok=True)
