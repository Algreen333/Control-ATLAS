import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

import cv2
from pymavlink import mavutil

from ZDC_vision_libs.camera import Camera


def survey_points(north, east, down, spacing, row_gap):
    for row in range(3):
        columns = range(5) if row % 2 == 0 else range(4, -1, -1)
        for column in columns:
            yield north + row * row_gap, east + column * spacing, down


def check_heartbeat(heartbeat):
    armed = heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    if not armed or mavutil.mode_string_v10(heartbeat) != "GUIDED":
        raise RuntimeError("El dron debe estar armado y en GUIDED.")


def wait_at_point(link, target, minimum_height=2.0, timeout=25):
    deadline = time.monotonic() + timeout
    heartbeat_at = time.monotonic()
    stable_samples = 0
    while time.monotonic() < deadline:
        message = link.recv_match(
            type=["HEARTBEAT", "LOCAL_POSITION_NED", "GLOBAL_POSITION_INT"],
            blocking=True,
            timeout=0.25,
        )
        if time.monotonic() - heartbeat_at > 10:
            raise RuntimeError("Se perdió el HEARTBEAT de SITL.")
        if message is None or (message.get_srcSystem(), message.get_srcComponent()) != (
            link.target_system, link.target_component
        ):
            continue

        kind = message.get_type()
        if kind == "HEARTBEAT":
            check_heartbeat(message)
            heartbeat_at = time.monotonic()
        elif kind == "GLOBAL_POSITION_INT":
            if message.relative_alt / 1000 < minimum_height:
                raise RuntimeError("El dron está demasiado bajo para el barrido.")
        elif target is None:
            return message
        else:
            error = math.dist((message.x, message.y, message.z), target)
            speed = math.hypot(message.vx, message.vy, message.vz)
            stable_samples = stable_samples + 1 if error <= 0.35 and speed <= 0.3 else 0
            if stable_samples >= 3:
                return message

    raise RuntimeError("No llegó una posición estable dentro del tiempo de espera.")


def capture_frame(camera, previous_timestamp):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        ok, frame, timestamp = camera.get_frame(copy=True)
        if ok and frame is not None and timestamp != previous_timestamp:
            return frame, timestamp
        time.sleep(0.05)
    raise RuntimeError("La cámara de Gazebo no entregó un fotograma nuevo.")


def main():
    parser = argparse.ArgumentParser(description='Barrido fotográfico de prueba para Control-ATLAS en Gazebo/SITL.')
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--row-gap", type=float, default=2.0)
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.spacing, args.row_gap)):
        parser.error("Las separaciones deben ser números positivos.")

    config = json.loads(Path("mission_config.json").read_text())
    vision = config["vision"]
    camera = Camera(
        source=vision.get("capture_source", 0),
        resolution=tuple(vision.get("capture_resolution", [640, 480])),
        fps=vision.get("capture_fps", 30),
        do_gazebo=True,
    )
    link = mavutil.mavlink_connection("udpin:127.0.0.1:14570", source_system=253)
    heartbeat = link.wait_heartbeat(timeout=10)
    if heartbeat is None:
        raise RuntimeError("Sin HEARTBEAT. Ejecuta en MAVProxy: output add 127.0.0.1:14570")
    check_heartbeat(heartbeat)
    link.target_system = heartbeat.get_srcSystem()
    link.target_component = heartbeat.get_srcComponent()

    for message_id in (32, 33):
        link.mav.command_long_send(
            link.target_system, link.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, message_id, 200_000, 0, 0, 0, 0, 0,
        )
    altitude = link.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
    if altitude is None or altitude.relative_alt < 2000:
        raise RuntimeError("Primero despega en GUIDED y estabiliza el dron a 5 m.")
    origin = wait_at_point(link, None)
    _, timestamp = capture_frame(camera, None)

    session = Path(mkdtemp(prefix="atlas-barrido-", dir=Path.home()))
    images = session / "images"
    images.mkdir()
    print(f"Sesión: {session}", flush=True)
    points = survey_points(origin.x, origin.y, origin.z, args.spacing, args.row_gap)
    with (session / "frames.csv").open("w", newline="") as metadata:
        writer = csv.writer(metadata)
        writer.writerow(["file", "saved_utc", "north_m", "east_m", "down_m"])
        for index, target in enumerate(points, start=1):
            link.mav.set_position_target_local_ned_send(
                0, link.target_system, link.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, 3576, *target,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
            position = wait_at_point(link, target)
            frame, timestamp = capture_frame(camera, timestamp)
            name = f"foto_{index:03d}.jpg"
            if not cv2.imwrite(str(images / name), frame):
                raise RuntimeError(f"No se pudo guardar {name}.")
            writer.writerow([
                name, datetime.now(timezone.utc).isoformat(),
                position.x, position.y, position.z,
            ])
            metadata.flush()
            print(f"{index}/15: {name}", flush=True)

    print("Barrido completado. Puedes aterrizar desde MAVProxy.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"Barrido detenido: {exc}") from exc
