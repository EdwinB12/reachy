"""Workaround for a Reachy Mini SDK bug affecting wired-network setups.

The daemon reports `wlan_ip` by reading the robot's wlan0 interface directly
(reachy_mini/daemon/utils.py:get_ip_address). When wlan0 is sitting in
hotspot mode (never joined to a real WiFi network) that IP is the hotspot's
own gateway (e.g. 10.42.0.1), which is not reachable from a client that talks
to the robot over Ethernet/mDNS. The SDK then uses that unreachable address
as the WebRTC signalling host, which hangs/fails even though the signalling
server is actually listening on every interface, including the one already
used for control-plane traffic.

Importing this module (before constructing ReachyMini) patches WSClient to
report the same host used for the control connection instead of the
possibly-wrong wlan_ip, so the SDK's default ("auto") media_backend works
without needing media_backend="no_media".
"""

import reachy_mini.io.ws_client as _ws_client_mod

_orig_get_status = _ws_client_mod.WSClient.get_status


def _get_status_with_fixed_wlan_ip(self, wait: bool = True, timeout: float = 5.0):
    status = _orig_get_status(self, wait=wait, timeout=timeout)
    status.wlan_ip = self.host
    return status


_ws_client_mod.WSClient.get_status = _get_status_with_fixed_wlan_ip
