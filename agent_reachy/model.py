import json
import os
from pathlib import Path

from langchain_ollama import ChatOllama

DEFAULT_MODEL = os.environ.get("REACHY_OLLAMA_MODEL", "qwen3.8:27b")
DEFAULT_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "https://llm.hpc.shef.ac.uk/ollama/"
)
API_KEY_PATH = (
    Path(__file__).resolve().parent.parent / ".secrets" / "ollama_api_key.json"
)


def _load_api_key() -> str:
    with open(API_KEY_PATH) as f:
        return json.load(f)["api_key"]


def Model(
    model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, **kwargs
) -> ChatOllama:
    kwargs.setdefault(
        "client_kwargs",
        {"headers": {"Authorization": f"Bearer {_load_api_key()}"}},
    )
    kwargs.setdefault("reasoning", True)
    return ChatOllama(model=model, base_url=base_url, **kwargs)
