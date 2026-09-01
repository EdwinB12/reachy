import json
import urllib.error
import urllib.request

BASE_URL = "http://reachy-mini.local:8000"


def _request(path: str, method: str = "GET") -> dict:
    request = urllib.request.Request(f"{BASE_URL}{path}", method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach {BASE_URL} ({e}). Is the robot on and reachable?"
        ) from e


def status() -> None:
    print(json.dumps(_request("/api/daemon/status"), indent=2))


def wake() -> None:
    print(json.dumps(_request("/api/daemon/start?wake_up=true", method="POST"), indent=2))


def sleep() -> None:
    print(json.dumps(_request("/api/daemon/stop?goto_sleep=true", method="POST"), indent=2))
