import json
import socket
import time
from typing import Optional, Tuple


class MatlabUdpBridge:
    """Small non-blocking UDP bridge between Python control and MATLAB."""

    def __init__(
        self,
        send_enabled: bool = False,
        host: str = "127.0.0.1",
        port: int = 15000,
        send_hz: float = 10.0,
        control_enabled: bool = False,
        control_port: int = 15001,
        command_timeout_s: float = 0.75,
        debug: bool = False,
    ):
        self.send_enabled = send_enabled
        self.host = host
        self.port = port
        self.control_enabled = control_enabled
        self.control_port = control_port
        self.command_timeout_s = command_timeout_s
        self.debug = debug

        self._send_period = 1.0 / max(send_hz, 0.001)
        self._last_send_t = 0.0
        self._seq = 0

        self._send_sock: Optional[socket.socket] = None
        self._control_sock: Optional[socket.socket] = None
        self._latest_command = None
        self._latest_command_t = 0.0

        if self.send_enabled:
            self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[MATLAB] UDP state sender -> {self.host}:{self.port}")

        if self.control_enabled:
            self._control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._control_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._control_sock.bind(("0.0.0.0", self.control_port))
            self._control_sock.setblocking(False)
            print(f"[MATLAB] UDP control receiver <- 0.0.0.0:{self.control_port}")

    def send_state(
        self,
        phase: str,
        visible: bool,
        err_x_m: Optional[float] = None,
        err_y_m: Optional[float] = None,
        dist_m: Optional[float] = None,
        alt_m: Optional[float] = None,
    ) -> None:
        """Send one JSON state packet to MATLAB, throttled by send_hz."""
        self.poll_command()

        if not self.send_enabled or self._send_sock is None:
            return

        now = time.monotonic()
        if now - self._last_send_t < self._send_period:
            return

        self._seq += 1
        payload = {
            "seq": self._seq,
            "timestamp": time.time(),
            "visible": bool(visible),
            "err_x_m": err_x_m,
            "err_y_m": err_y_m,
            "dist_m": dist_m,
            "alt_m": alt_m,
            "phase": phase,
        }

        try:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._send_sock.sendto(data, (self.host, self.port))
            self._last_send_t = now
            if self.debug:
                print(f"[MATLAB TX] {payload}")
        except OSError as exc:
            print(f"[MATLAB] WARNING: failed to send UDP state: {exc}")

    def poll_command(self) -> None:
        """Drain all pending MATLAB command datagrams without blocking."""
        if not self.control_enabled or self._control_sock is None:
            return

        while True:
            try:
                data, addr = self._control_sock.recvfrom(4096)
            except BlockingIOError:
                return
            except OSError as exc:
                print(f"[MATLAB] WARNING: failed to receive UDP command: {exc}")
                return

            try:
                raw = json.loads(data.decode("utf-8"))
                command = self._normalize_command(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                print(f"[MATLAB] WARNING: invalid UDP command from {addr}: {exc}")
                continue

            if command is None:
                continue

            if not command["valid"]:
                self._latest_command = None
                self._latest_command_t = 0.0
                if self.debug:
                    print(f"[MATLAB RX] invalid command from {addr}; using Python control")
                continue

            self._latest_command = command
            self._latest_command_t = time.monotonic()
            if self.debug:
                print(f"[MATLAB RX] {command}")

    def get_velocity_override(self) -> Optional[Tuple[float, float, float]]:
        """Return MATLAB body-frame velocity if fresh, otherwise None."""
        self.poll_command()

        if self._latest_command is None:
            return None

        age = time.monotonic() - self._latest_command_t
        if age > self.command_timeout_s:
            if self.debug:
                print(f"[MATLAB] command stale ({age:.2f}s); using Python control")
            self._latest_command = None
            self._latest_command_t = 0.0
            return None

        return (
            self._latest_command["vx"],
            self._latest_command["vy"],
            self._latest_command["vz"],
        )

    def close(self) -> None:
        for sock in (self._send_sock, self._control_sock):
            if sock is not None:
                sock.close()

    @staticmethod
    def _normalize_command(raw):
        if not isinstance(raw, dict):
            raise ValueError("command must be a JSON object")

        valid = bool(raw.get("valid", True))
        if not valid:
            return {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0, "valid": False}

        return {
            "vx": float(raw.get("vx", 0.0)),
            "vy": float(raw.get("vy", 0.0)),
            "vz": float(raw.get("vz", 0.0)),
            "yaw_rate": float(raw.get("yaw_rate", 0.0)),
            "valid": True,
        }
