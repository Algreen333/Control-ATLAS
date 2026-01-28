from pymavlink import mavutil
import cv2
from cv2 import aruco

from lib.movement import *
from lib.pose_estimation import *

import time
import math

import os


DELAY_TIME = 5
AVG_COUNT = 10

### INICIALITZAR CAMERA I DETECTOR ARUCOS
# Open GSTREAMER pipeline for camera
VIDSRC_PORT = 5600

pipeline = (
    f"udpsrc port={VIDSRC_PORT} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

mtx = get_gazebo_camera_matrix(640, 480, 2)
dst = None

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)


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

        vid_count = 0
        last_update = 0
        xs = []
        ys = []
        zs = []

        inp = cv2.waitKey(30)
        enabled = False
        recording = False
        while inp != ord('q'):
            ret, frame = cap.read()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # corners: A list of arrays. Each item corresponds to one detected marker.
                # ids: A list of IDs corresponding to the corners.
                corners, ids, rejected = detector.detectMarkers(gray)

                if ids is not None:
                    for i in range(len(ids)):
                        try:
                            x, y, z = get_aruco_offset(mtx, dst, corners[i])

                            if (last_update + DELAY_TIME < time.time() and enabled):
                                if (len(xs) < AVG_COUNT):
                                    xs.append(x)
                                    ys.append(y)
                                    zs.append(z)
                                else:
                                    x, y, z = average_pos(xs, ys, zs)
                                    xs = []
                                    ys = []
                                    zs = []

                                    dist = math.sqrt(x*x + y*y)
                                    if (z < 3):
                                        if (dist < 0.15):
                                            print("ATERRANT...")
                                            master.mav.command_long_send(
                                                master.target_system, master.target_component,
                                                mavutil.mavlink.MAV_CMD_NAV_LAND,
                                                0, 0, 0, 0, 0, 0, 0, 0)
                                        else:
                                            move_rel(master, -y, x, 0)
                                            print("MOVENT:", -y, x, 0)
                                    elif (dist < 2):
                                        move_rel(master, -y, x, 4)
                                        print("MOVENT:", -y, x, 4)

                                    else:
                                        move_rel(master, -y, x, 0)
                                        print("MOVENT:", -y, x, 0)



                        except Exception as e: print(e)

                        aruco.drawDetectedMarkers(frame, corners, ids)
                elif enabled: print("None found")

                cv2.imshow("frame", frame)

                if recording:
                    writer.write(frame)

            inp_prev = inp
            inp = cv2.waitKey(30)
            if (inp != inp_prev and inp == ord('s')):
                enabled = not enabled
                if enabled:
                    esperar_trigger_inici() 
                    print("enabled")
                else: print("disabled")
            if (inp != inp_prev and inp == ord('r')):
                recording = not recording
                if recording:
                    while os.path.exists(f"output_{vid_count}.mp4"): vid_count += 1
                    ### VIDEO WRITER
                    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                    writer = cv2.VideoWriter(f"output_{vid_count}.mp4", fourcc, 10.0, (640, 480))
                    print("recording")
                else: 
                    print("not recording")
                    writer.release()

        cap.release()
        try: writer.release()
        except: pass

    # S'executa si s'interromp l'execució del programa
    finally:
        try: writer.release()
        except: pass
        cap.release()