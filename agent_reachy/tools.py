import time
from typing import Literal

import numpy as np
import requests
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool

import agent_reachy.reachy_mini_fix as reachy_mini_fix  # noqa: F401 (patches wlan_ip detection before ReachyMini connects)
from agent_reachy.daemon import BASE_URL
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.recorded_move import RecordedMoves

EMOTIONS_LIB = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

# Built from the loaded library rather than hardcoded, so the allowed values
# can never drift from what's actually available to play.
EMOTIONS = Literal[tuple(EMOTIONS_LIB.list_moves())]


RED = "\033[31m"
RESET = "\033[0m"


class ToolLoggingHandler(BaseCallbackHandler):
    """Prints to the terminal whenever a tool starts and finishes."""

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        print(f"\n{RED}[tool] {name}({input_str}){RESET}")


def build_toolbox(mini: ReachyMini) -> list:
    """Build the agent's toolbox, bound to a single shared ReachyMini connection."""

    @tool
    def wake_up() -> str:
        """Wake Reachy Mini up: head to neutral pose, plays its wake-up sound and emote.

        Returns:
            A short confirmation message.
        """
        mini.enable_motors()
        mini.wake_up()
        return "Reachy Mini is awake."

    @tool
    def go_to_sleep() -> str:
        """Put Reachy Mini to sleep: head and antennas fold into the sleep pose, plays its sleep sound.

        Returns:
            A short confirmation message.
        """
        mini.goto_sleep()
        mini.disable_motors()
        return "Reachy Mini is asleep."

    @tool
    def display_emotion(emotion: EMOTIONS) -> str:
        """Display a recorded emotion from the Reachy Mini Emotions Library.

        Args:
            emotion: The name of the emotion to display. Must be one of the
                predefined emotions in the library (see the allowed values
                in this argument's schema).

        Returns:
            A short confirmation message.
        """
        mini.play_move(EMOTIONS_LIB.get(emotion))
        return f"Displaying emotion: {emotion}"

    @tool
    def move_head(
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_mm: float = 0.0,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        duration: float = 1.0,
    ) -> str:
        """Move Reachy Mini's head to a target pose relative to neutral.

        The head stays at the target pose until moved again (it does not return
        to neutral on its own).

        Args:
            x_mm: Forward/backward head displacement in millimetres (positive forward).
            y_mm: Left/right head displacement in millimetres (positive left).
            z_mm: Vertical head displacement in millimetres (positive up).
            roll_deg: Head roll (tilt side to side) in degrees.
            pitch_deg: Head pitch (nod up/down) in degrees.
            yaw_deg: Head yaw (turn left/right) in degrees.
            duration: Time in seconds for the movement. Defaults to 1.0.

        Returns:
            A short confirmation message once the movement completes.
        """
        pose = create_head_pose(
            x=x_mm,
            y=y_mm,
            z=z_mm,
            roll=roll_deg,
            pitch=pitch_deg,
            yaw=yaw_deg,
            mm=True,
        )
        mini.goto_target(head=pose, duration=duration, body_yaw=None)
        return f"Moved head (x={x_mm}mm, y={y_mm}mm, z={z_mm}mm, roll={roll_deg}deg, pitch={pitch_deg}deg, yaw={yaw_deg}deg)."

    @tool
    def move_antennas(
        right_deg: float, left_deg: float, duration: float = 1.0
    ) -> str:
        """Move Reachy Mini's two antennas to target angles.

        The antennas stay at the target angles until moved again.

        Args:
            right_deg: Target angle of the right antenna in degrees.
            left_deg: Target angle of the left antenna in degrees.
            duration: Time in seconds for the movement. Defaults to 1.0.

        Returns:
            A short confirmation message once the movement completes.
        """
        mini.goto_target(
            antennas=np.deg2rad([right_deg, left_deg]),
            duration=duration,
            body_yaw=None,
        )
        return f"Moved antennas (right={right_deg}deg, left={left_deg}deg)."

    @tool
    def move_body(yaw_deg: float, duration: float = 1.0) -> str:
        """Rotate Reachy Mini's body to a target yaw angle.

        The body stays at the target angle until moved again.

        Args:
            yaw_deg: Target body rotation in degrees (positive turns one way, negative the other).
            duration: Time in seconds for the movement. Defaults to 1.0.

        Returns:
            A short confirmation message once the movement completes.
        """
        mini.goto_target(body_yaw=np.deg2rad(yaw_deg), duration=duration)
        return f"Rotated body to yaw={yaw_deg}deg."

    @tool
    def look_at(x_m: float, y_m: float, z_m: float, duration: float = 1.0) -> str:
        """Make Reachy Mini's head look at a point in 3D space.

        The frame is centred on the neutral head position: x forward, y left, z up, in metres.

        Args:
            x_m: Forward distance to the point, in metres.
            y_m: Left/right distance to the point, in metres (positive left).
            z_m: Height of the point, in metres (positive up).
            duration: Time in seconds for the movement. Defaults to 1.0.

        Returns:
            A short confirmation message once the movement completes.
        """
        mini.look_at_world(x_m, y_m, z_m, duration=duration)
        return f"Looked at point (x={x_m}m, y={y_m}m, z={z_m}m)."

    @tool
    def detect_face() -> str:
        """Check whether Reachy Mini's camera currently sees a face, and where.

        Briefly starts head-tracking, takes one snapshot reading, then stops
        tracking again. Useful for answering questions like "is anyone looking
        at you?" or "where is the person relative to your camera?".

        Returns:
            A message stating whether a face was detected, and if so its
            normalized x/y position in the camera frame (roughly -1.0 to 1.0,
            where 0.0, 0.0 is the center of the frame).
        """
        mini.start_head_tracking()
        time.sleep(2)
        try:
            face = mini.get_tracked_face()
        finally:
            mini.stop_head_tracking()

        if face.detected:
            return f"Face detected at x={face.x:+.2f}, y={face.y:+.2f} (normalized camera coordinates)."
        return "No face detected."

    @tool
    def read_imu() -> str:
        """Read Reachy Mini's IMU (Inertial Measurement Unit) sensors.

        Reports the current accelerometer, gyroscope, orientation quaternion, and
        internal temperature. Use this to check whether the robot is being moved,
        tilted, or shaken, or to check its thermal state.

        Returns:
            A human-readable summary of the accelerometer (m/s^2), gyroscope
            (rad/s), orientation quaternion (w, x, y, z), and temperature (deg C).
        """
        imu_data = mini.imu

        accel_x, accel_y, accel_z = imu_data["accelerometer"]
        gyro_x, gyro_y, gyro_z = imu_data["gyroscope"]
        quat_w, quat_x, quat_y, quat_z = imu_data["quaternion"]
        temperature = imu_data["temperature"]

        return (
            f"Accelerometer (m/s^2): x={accel_x:.3f}, y={accel_y:.3f}, z={accel_z:.3f}\n"
            f"Gyroscope (rad/s): x={gyro_x:.3f}, y={gyro_y:.3f}, z={gyro_z:.3f}\n"
            f"Quaternion: w={quat_w:.3f}, x={quat_x:.3f}, y={quat_y:.3f}, z={quat_z:.3f}\n"
            f"Temperature: {temperature:.1f} deg C"
        )

    @tool
    def record_and_playback_audio(listen_seconds: float = 3.0) -> str:
        """Record audio from Reachy Mini's microphone for a few seconds, then play it back.

        Useful for testing the robot's microphone and speaker, or for a simple
        "echo" interaction where the robot repeats back what it just heard. While
        this tool is running, both audio devices are exclusively held by the robot
        and unavailable to other applications.

        Args:
            listen_seconds: How many seconds of audio to record before playing it
                back. Defaults to 3.0.

        Returns:
            A message describing how much audio was captured and its peak volume,
            or a message noting that no audio was captured.
        """
        mini.media.start_recording()
        mini.media.start_playing()
        time.sleep(2)  # Wait for the audio devices to be ready

        try:
            # Recording runs continuously in the background, so discard any
            # backlog before the timed listen window starts.
            flush_until = time.monotonic() + 0.2
            while time.monotonic() < flush_until:
                mini.media.get_audio_sample()

            chunks = []
            start = time.monotonic()
            while time.monotonic() - start < listen_seconds:
                chunk = mini.media.get_audio_sample()
                if chunk is not None:
                    chunks.append(chunk)

            if not chunks:
                return "No audio captured, nothing to play back."

            recording = np.concatenate(chunks, axis=0)
            duration = len(recording) / mini.media.get_input_audio_samplerate()
            peak = float(np.max(np.abs(recording)))

            mini.media.push_audio_sample(recording)
            time.sleep(len(recording) / mini.media.get_output_audio_samplerate())

            return f"Captured {duration:.2f}s of audio (peak level {peak:.4f}) and played it back."
        finally:
            mini.media.stop_recording()
            mini.media.stop_playing()

    @tool
    def get_volume() -> str:
        """
        Get the current volume of Reachy Mini's speaker.

        Returns:
            A message stating the current volume level, between 0 (mute) and 100 (max).
        """
        response = requests.get(f"{BASE_URL}/api/volume/current")
        response.raise_for_status()
        return f"Current volume is {response.json()['volume']}."

    @tool
    def change_volume(volume: int):
        """
        Change the volume of Reachy Mini's speaker.

        Args:
            volume: The desired volume level, between 0 (mute) and 100 (max).
        """
        response = requests.post(
            f"{BASE_URL}/api/volume/set", json={"volume": volume}
        )
        response.raise_for_status()
        return f"Volume set to {response.json()['volume']}."

    return [
        wake_up,
        go_to_sleep,
        move_head,
        move_antennas,
        move_body,
        look_at,
        display_emotion,
        detect_face,
        read_imu,
        record_and_playback_audio,
        get_volume,
        change_volume,
    ]
