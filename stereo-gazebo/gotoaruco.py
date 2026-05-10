from pymavlink import mavutil
import cv2
from scipy.spatial.transform import Rotation, Slerp


from lib.movement import *
from lib.pose_estimation import *
from lib.aruco_lib import *

import time
import math

import sys
import os

DELAY_TIME = 5
AVG_COUNT = 10

### INICIALITZAR CAMERA I DETECTOR ARUCOS
# Open GSTREAMER pipeline for camera
VIDSRC_PORT_WIDE = 5800
VIDSRC_PORT_NARR = 5700

pipeline_wide = (
    f"udpsrc port={VIDSRC_PORT_WIDE} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=1"
)
pipeline_narr = (
    f"udpsrc port={VIDSRC_PORT_NARR} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=1"
)


cap_wide = cv2.VideoCapture(pipeline_wide, cv2.CAP_GSTREAMER)
print(cap_wide.isOpened())

cap_narr = cv2.VideoCapture(pipeline_narr, cv2.CAP_GSTREAMER)
print(cap_wide.isOpened())


mtx_wide = get_gazebo_camera_matrix(1536, 864, 102/180*np.pi, 48.8/180*np.pi)
dst_wide = np.zeros(5, dtype=np.float32)
mtx_narr = get_gazebo_camera_matrix(1640, 1232, 62.2/180*np.pi, 67/180*np.pi)
dst_narr = np.zeros(5, dtype=np.float32)

detWide = ArucoDetector(mtx_wide, dst_wide, cv2.aruco.DICT_4X4_50)
detNarr = ArucoDetector(mtx_wide, dst_wide, cv2.aruco.DICT_4X4_50)

T_C_WIDE = np.eye(4)
T_C_WIDE[0, 3] = 50
T_C_NARR = np.eye(4)
T_C_NARR[0, 3] = -50

SERVER_ID = 0
CLIENT_ID = 1

DRAW_ARUCOS = True


### INICIALITZAR MAVLINK
connection_string = 'tcp:localhost:5762' #port intern del ordinador | revisa que no estigui ocupat ja
master = mavutil.mavlink_connection(connection_string)
master.wait_heartbeat()

# Obtenim els IDs dels modes necessaris
mode_guided_id = master.mode_mapping()['GUIDED']
mode_land_id = master.mode_mapping()['LAND']



## ESPERA A INICI
def esperar_trigger_inici():
    """
    Bucle espera, comprovar abans de passar autònom:
        - Motors ARMATS
        - Mode GUIDED
    """
    print(">>> Esperant GUIDED o ARMAT...")
    
    while True:
        msg = master.recv_match(type='HEARTBEAT', blocking=True)
        if msg:
            esta_armat = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            es_guided = (msg.custom_mode == mode_guided_id)
            
            if esta_armat and es_guided:
                print(">>> Dron preparat: Començant missió")
                # Buidem buffer per seguretat abans de començar
                while master.recv_match(blocking=False): pass
                return

def esperar_reset_manual():
    """ 
    Espera que es tregui mode GUIDED
    """
    print("Treu mode GUIDED per poder seguir")
    
    while True:
        msg = master.recv_match(type='HEARTBEAT', blocking=True)
        if msg:
            if msg.custom_mode != mode_guided_id:
                print(">>> Reset completat")
                return

def get_aruco_offset(mtx, dst, corner):
    rvec, tvec = estimate_pose(corner, 1, mtx, dst)
    distance = np.linalg.norm(tvec)
    print(f"Distance: {distance:.3f} meters")
    x_offset = tvec[0][0]
    y_offset = tvec[1][0]
    z_depth  = tvec[2][0]
    print(f"Position: x={x_offset:.2f}, y={y_offset:.2f}, z={z_depth:.2f}")
    pitch, yaw, roll = get_euler_angles(rvec)
    print(f"Orientation: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}")
    return (x_offset, y_offset, z_depth)

def average_pos(xs, ys, zs):
    x = 0
    for i in xs: x+=i
    x /= len(xs)
    
    y = 0
    for i in ys: y+=i
    y /= len(ys)

    z = 0
    for i in zs: z+=i
    z /= len(zs)

    return x, y, z



if __name__ == "__main__":
    try:
        inp = cv2.waitKey(30)

        print("start")
        enabled = False
        recording = False
        while inp != ord('q'):
            ret_w, frame_wide = cap_wide.read()
            ret_n, frame_narr = cap_narr.read()

            if (not ret_w) or (not ret_n):
                print("err")
                time.sleep(0.1)
                continue

            gray_wide = cv2.cvtColor(frame_wide, cv2.COLOR_BGR2GRAY)
            corners_wide, ids_wide, rejected_wide = detWide.detectMarkers(gray_wide)
            
            gray_narr = cv2.cvtColor(frame_narr, cv2.COLOR_BGR2GRAY)
            corners_narr, ids_narr, rejected_narr = detNarr.detectMarkers(gray_narr)

            if ids_wide is not None and ids_narr is not None:
                rvec_wide, tvec_wide = detWide.estimate_pose(corners_wide[0], 180)
                rvec_narr, tvec_narr = detNarr.estimate_pose(corners_narr[0], 180)
                rvec, tvec = fuse_stereo_aruco_poses(rvec_wide, tvec_wide, T_C_WIDE, rvec_narr, tvec_narr, T_C_NARR)
                
                
                output = (
                    f"----------------RVEC----------------\n"
                    f"{rvec}\n"
                    f"----------------TVEC----------------\n"
                    f"{tvec}\n"
                )
                sys.stdout.write(output)
                sys.stdout.flush()

                sys.stdout.write("\033[F" * 8)

            if SERVER_ID==0: frame_to_encode = joint_display(frame_wide, frame_narr, 0)
            else: frame_to_encode = joint_display(frame_narr, frame_wide, 0)

            cv2.imshow("frame", frame_to_encode)
            inp = cv2.waitKey(30)
    except Exception as e:
        print("Exception:", e)
        pass

