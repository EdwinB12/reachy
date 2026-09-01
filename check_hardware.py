from reachy_mini.daemon.app.routers.volume_control import get_volume_control

import agent_reachy.reachy_mini_fix as reachy_mini_fix  # noqa: F401 (patches wlan_ip detection before ReachyMini connects)

import logging

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
import requests
import time

logging.basicConfig(level=logging.INFO)


def basic_movement(mini: ReachyMini):
    mini.goto_target(
        head=create_head_pose(z=10, mm=True),  # Up 10mm
        antennas=np.deg2rad([45, 45]),  # Antennas out
        body_yaw=np.deg2rad(30),  # Turn body
        duration=2.0,  # Take 2 seconds
        method="minjerk",  # Smooth acceleration
    )

    mini.goto_target(
        head=create_head_pose(z=-10, mm=True),  # Up 10mm
        antennas=np.deg2rad([-45, -45]),  # Antennas out
        body_yaw=np.deg2rad(-30),  # Turn body
        duration=2.0,  # Take 2 seconds
        method="minjerk",  # Smooth acceleration
    )

    mini.goto_target(
        head=create_head_pose(z=0, mm=True),  # Up 10mm
        antennas=np.deg2rad([0, 0]),  # Antennas out
        body_yaw=np.deg2rad(0),  # Turn body
        duration=2.0,  # Take 2 seconds
        method="minjerk",  # Smooth acceleration
    )


def test_headtracking(mini: ReachyMini):
    """
    Test head tracking for 10 seconds
    """

    timeout = 10  # seconds
    start = time.monotonic()

    mini.start_head_tracking()
    try:
        while time.monotonic() - start < timeout:
            face = mini.get_tracked_face()
            if face.detected:
                print(f"Face at x={face.x:+.2f}, y={face.y:+.2f}")
            else:
                print("No face detected")
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping head tracking")
        mini.stop_head_tracking()


def test_imu(mini: ReachyMini):
    """
    Test the Accelerometer, Gyroscope, quaternion and temperature sensors
    """
    imu_data = mini.imu
    print(f"Keys in imu_data: {list(imu_data.keys())}")

    accel_x, accel_y, accel_z = imu_data["accelerometer"]  # (m/s^2)
    print(f"Accelerometer data: x={accel_x}, y={accel_y}, z={accel_z}")

    gyro_x, gyro_y, gyro_z = imu_data["gyroscope"]  # (rad/s)
    print(f"Gyroscope data: x={gyro_x}, y={gyro_y}, z={gyro_z}")

    quat_w, quat_x, quat_y, quat_z = imu_data["quaternion"]  # (w, x, y, z)
    print(f"Quaternion data: w={quat_w}, x={quat_x}, y={quat_y}, z={quat_z}")

    temperature = imu_data["temperature"]  # (°C)
    print(f"Temperature data: {temperature}°C")


def test_audio_playback(mini: ReachyMini):
    """
    Listen for 3 seconds, then play back what was recorded.
    """
    listen_duration = 3.0  # seconds

    # Initialization - After this point, both audio devices (input/output) will be seen as busy by other applications!
    mini.media.start_recording()
    mini.media.start_playing()
    time.sleep(2)  # Wait for the audio devices to be ready

    try:
        # Recording runs continuously in the background (it started as soon as
        # the connection opened), so the queue may already hold a backlog of
        # older audio. Discard samples for a short window so the timed listen
        # below only captures audio from this point forward. (New audio keeps
        # arriving in real time, so we can't just loop "until empty".)
        flush_until = time.monotonic() + 0.2
        while time.monotonic() < flush_until:
            mini.media.get_audio_sample()

        print(f"Listening for {listen_duration:.0f} seconds...")
        chunks = []
        start = time.monotonic()
        while time.monotonic() - start < listen_duration:
            chunk = mini.media.get_audio_sample()
            if chunk is not None:
                chunks.append(chunk)

        if not chunks:
            print("No audio captured, nothing to play back.")
            return

        recording = np.concatenate(chunks, axis=0)
        duration = len(recording) / mini.media.get_input_audio_samplerate()
        peak = np.max(np.abs(recording))
        print(
            f"Captured {duration:.2f}s of audio (peak level {peak:.4f}), playing it back..."
        )

        # Play
        mini.media.push_audio_sample(recording)
        time.sleep(len(recording) / mini.media.get_output_audio_samplerate())
    except Exception as e:
        print(f"Error occurred while testing audio playback: {e}")
    finally:
        # Release audio devices (input/output)
        mini.media.stop_recording()
        mini.media.stop_playing()


def run():

    with ReachyMini() as mini:
        mini.wake_up()

        time.sleep(2)

        # Test some movement
        try:
            basic_movement(mini)
        except Exception as e:
            print(f"Error occurred while testing basic movement: {e}")

        # Perform head tracking
        try:
            test_headtracking(mini)
        except Exception as e:
            print(f"Error occurred while testing head tracking: {e}")

        # Test IMU readings
        try:
            test_imu(mini)
        except Exception as e:
            print(f"Error occurred while testing IMU readings: {e}")

        # Test audio playback (listen 3s, then play it back)
        try:
            test_audio_playback(mini)
        except Exception as e:
            print(f"Error occurred while testing audio playback: {e}")

        mini.goto_sleep()


if __name__ == "__main__":
    run()
