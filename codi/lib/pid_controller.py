"""
PID Controller for autonomous drone lateral correction during precision landing.

Each axis (forward, right) gets its own PIDController instance.
The controller includes:
  - Anti-windup via integral clamping
  - Derivative-on-measurement (not on error) to avoid derivative kick on setpoint changes
  - Low-pass filtered derivative term to reduce noise from vision measurements
  - Output saturation
  - Reset method for phase transitions
"""

import time


class PIDController:
    """Discrete PID controller with anti-windup and derivative filtering."""

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.0,
        kd: float = 0.1,
        setpoint: float = 0.0,
        output_min: float = -1.0,
        output_max: float = 1.0,
        integral_max: float = 0.5,
        derivative_filter_alpha: float = 0.3,
    ):
        """
        Parameters
        ----------
        Tuning steps:
        1. Set Ki=0, Kd=0. Increase Kp until it oscillates, then halve it.
        2. Increase Kd until oscillations are damped (try 0.1 → 0.3).
        3. Add small Ki (0.01 → 0.1) only if there's persistent steady-state offset.
        4. Tune descent-rate last: lower it if the drone loses the marker during descent.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        self.alpha = derivative_filter_alpha

        # Internal state
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_derivative = 0.0
        self._prev_time = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_derivative = 0.0
        self._prev_time = None

    def update(self, measurement: float, dt: float | None = None) -> float:
        """
        Compute the PID output for this timestep.

        Parameters
        ----------
        measurement : Current measured value (e.g. lateral offset in metres).
                      Positive measurement → target is to the right/forward.
        dt : Timestep in seconds. If None, computed from wall clock since last call.

        Returns
        -------
        Command velocity (m/s), clamped to [output_min, output_max].
        The sign convention is: positive output commands movement in the
        positive measurement direction (toward the target).
        """
        now = time.monotonic()

        if dt is None:
            if self._prev_time is None:
                dt = 0.0
            else:
                dt = now - self._prev_time
        self._prev_time = now

        # Prevent stale huge dt (e.g. after pause or target loss)
        dt = min(dt, 0.5)

        # Positive measurement (marker offset to the right/forward) →
        # positive error → positive output → fly toward the marker.
        error = measurement - self.setpoint

        # ── Proportional ──
        p_term = self.kp * error

        # ── Integral ── (trapezoidal would be overkill here; rectangular is fine at 10+ Hz)
        if dt > 0:
            self._integral += error * dt
            # Anti-windup clamp
            self._integral = max(-self.integral_max, min(self.integral_max, self._integral))
        i_term = self.ki * self._integral

        # ── Derivative on measurement ──
        # Using derivative-on-measurement avoids spikes when setpoint changes.
        if self._prev_measurement is not None and dt > 0:
            raw_derivative = (measurement - self._prev_measurement) / dt
            # Low-pass EMA filter
            filtered_derivative = (
                self.alpha * raw_derivative
                + (1.0 - self.alpha) * self._prev_derivative
            )
        else:
            filtered_derivative = 0.0

        self._prev_measurement = measurement
        self._prev_derivative = filtered_derivative
        d_term = self.kd * filtered_derivative

        # ── Sum and saturate ──
        output = p_term + i_term + d_term
        output = max(self.output_min, min(self.output_max, output))

        return output

    def __repr__(self) -> str:
        return (
            f"PIDController(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"integral={self._integral:.3f})"
        )