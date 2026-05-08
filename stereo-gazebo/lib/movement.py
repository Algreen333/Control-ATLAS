from pymavlink import mavutil

def move_rel(master, x, y, z):
    master.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(0, master.target_system, master.target_component, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 0b010111111000, x, y, z, 0, 0, 0, 0, 0, 0, 0, 0))