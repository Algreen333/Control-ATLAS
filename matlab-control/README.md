# MATLAB UDP control bridge

This folder contains a minimal UDP bridge for using MATLAB as a control/decision
layer while Python/OpenCV keeps detecting ArUco markers and Python/MAVLink keeps
commanding ArduPilot.

No absolute paths are required. Run the Python commands from
`stereo-gazebo-from-scratch/` or adjust the script path from the repo root.

## Ports

- Python -> MATLAB state: UDP `15000`
- MATLAB -> Python control: UDP `15001`

Change the MATLAB host with `--matlab-host` if MATLAB runs on another computer.
For example, use the IP address of the MATLAB machine instead of `127.0.0.1`.

## State JSON sent by Python

Python sends:

```json
{
  "seq": 1,
  "timestamp": 1780000000.0,
  "visible": true,
  "err_x_m": 0.12,
  "err_y_m": -0.08,
  "dist_m": 0.15,
  "alt_m": 4.8,
  "phase": "descent"
}
```

Field meanings:

- `seq`: packet sequence number.
- `timestamp`: Python wall-clock timestamp in seconds.
- `visible`: true when the ArUco pose is currently available.
- `err_x_m`: forward body-frame offset to the target in meters. Positive is forward.
- `err_y_m`: right body-frame offset to the target in meters. Positive is right.
- `dist_m`: horizontal distance `sqrt(err_x_m^2 + err_y_m^2)`.
- `alt_m`: MAVLink relative altitude when available in descent, otherwise the
  ArUco down/depth estimate. It can be `null` when no estimate is available.
- `phase`: `approach`, `descent`, or `final`.

## Command JSON expected by Python

MATLAB can send:

```json
{
  "vx": 0.0,
  "vy": 0.0,
  "vz": 0.05,
  "yaw_rate": 0.0,
  "valid": true
}
```

`vx`, `vy`, and `vz` are body-frame velocities in m/s using the same convention
as `move_velocity_body`: positive `vx` is forward, positive `vy` is right, and
positive `vz` is down. `yaw_rate` is accepted for future use but is currently
ignored by `main-no-plnd.py`.

If no valid command arrives before `--matlab-command-timeout` seconds, Python
falls back to its original controller.

## Run without MATLAB

This is unchanged:

```bash
cd stereo-gazebo-from-scratch
python3 main-no-plnd.py --gazebo --auto_arm --takeoff --alt 5 --display
```

## Python sends state to MATLAB

In MATLAB, run:

```matlab
cd matlab-control
matlab_udp_receiver
```

From another terminal:

```bash
cd stereo-gazebo-from-scratch
python3 main-no-plnd.py --gazebo --auto_arm --takeoff --alt 5 --display --matlab-udp --matlab-host 127.0.0.1 --matlab-port 15000
```

## Python sends state and accepts MATLAB control

In MATLAB, run:

```matlab
cd matlab-control
matlab_control_loop_demo
```

From another terminal:

```bash
cd stereo-gazebo-from-scratch
python3 main-no-plnd.py --gazebo --auto_arm --takeoff --alt 5 --display --matlab-udp --matlab-control
```

## MATLAB on another computer

Use the MATLAB computer IP in Python:

```bash
python3 main-no-plnd.py --gazebo --auto_arm --takeoff --alt 5 --display --matlab-udp --matlab-host 192.168.1.50 --matlab-port 15000
```

If MATLAB sends commands from another machine, make sure UDP `15001` can reach
the Python computer and that local firewalls allow both ports.

## MATLAB requirement

The scripts use `udpport`, available in recent MATLAB versions. Some MATLAB
installations/versions may require Instrument Control Toolbox for UDP support.
