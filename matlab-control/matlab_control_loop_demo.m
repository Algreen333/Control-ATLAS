% Demo control loop:
% 1) Receives ArUco/drone state JSON from Python on UDP 15000.
% 2) Sends body-frame velocity commands back to Python on UDP 15001.
%
% Run Python with:
% python3 main-no-plnd.py --gazebo --auto_arm --takeoff --alt 5 --display --matlab-udp --matlab-control

clear; clc;

statePort = 15000;
pythonHost = "127.0.0.1";
pythonControlPort = 15001;

kxy = 0.12;        % proportional gain for lateral control
maxLateral = 0.12; % m/s
descentVz = 0.025; % positive is down in Python body-NED velocity

u = udpport("datagram", "IPV4", "LocalPort", statePort);

fprintf("Listening for Python state on UDP %d\n", statePort);
fprintf("Sending MATLAB control to %s:%d\n", pythonHost, pythonControlPort);
fprintf("Stop with Ctrl+C.\n\n");

while true
    latestState = [];

    while u.NumDatagramsAvailable > 0
        msg = read(u, 1, "string");
        try
            latestState = jsondecode(char(msg));
        catch err
            fprintf("Invalid state packet: %s\n", err.message);
        end
    end

    if ~isempty(latestState)
        cmd = makeCommand(latestState, kxy, maxLateral, descentVz);
        write(u, jsonencode(cmd), "string", pythonHost, pythonControlPort);

        fprintf("[%s] visible=%d cmd=(%.3f, %.3f, %.3f) valid=%d\n", ...
            string(latestState.phase), latestState.visible, ...
            cmd.vx, cmd.vy, cmd.vz, cmd.valid);
    end

    pause(0.02);
end

function cmd = makeCommand(state, kxy, maxLateral, descentVz)
    if ~state.visible || isempty(state.err_x_m) || isempty(state.err_y_m)
        cmd = struct("vx", 0.0, "vy", 0.0, "vz", 0.0, ...
                     "yaw_rate", 0.0, "valid", false);
        return;
    end

    vx = clamp(kxy * double(state.err_x_m), -maxLateral, maxLateral);
    vy = clamp(kxy * double(state.err_y_m), -maxLateral, maxLateral);

    if strcmp(string(state.phase), "descent")
        vz = descentVz;
    else
        vz = 0.0;
    end

    cmd = struct("vx", vx, "vy", vy, "vz", vz, ...
                 "yaw_rate", 0.0, "valid", true);
end

function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end
