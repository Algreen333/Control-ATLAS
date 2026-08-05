% Receive ArUco/drone state JSON packets from Python and print them.
% Run this before starting Python with --matlab-udp.

clear; clc;

localPort = 15000;
u = udpport("datagram", "IPV4", "LocalPort", localPort);

fprintf("Listening for Python state packets on UDP port %d...\n", localPort);
fprintf("Stop with Ctrl+C.\n\n");

while true
    while u.NumDatagramsAvailable > 0
        msg = read(u, 1, "string");

        try
            state = jsondecode(char(msg));
            fprintf("[%s] seq=%d visible=%d err=(%.3f, %.3f) dist=%.3f alt=%.3f\n", ...
                string(state.phase), state.seq, state.visible, ...
                valueOrNaN(state.err_x_m), valueOrNaN(state.err_y_m), ...
                valueOrNaN(state.dist_m), valueOrNaN(state.alt_m));
        catch err
            fprintf("Invalid packet: %s\n", err.message);
            disp(msg);
        end
    end

    pause(0.02);
end

function value = valueOrNaN(raw)
    if isempty(raw)
        value = NaN;
    else
        value = double(raw);
    end
end
