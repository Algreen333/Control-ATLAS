import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (main)
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lib.camera_lib import *
from lib.aruco_lib import *

cap = VideoCapture(format="RGB888", resolution=(1536, 864))
cap.start()

conf = CalibrationConfig.from_path("../configs/1536x864.conf")

aruco_detector = ArucoDetector(conf.mtx, conf.dist)

def processor(frame):
    ret = aruco_detector.full_prediction(frame)
    time.sleep(0.1)
    if ret is not None: 
        new_frame, pos, rot = ret
        print("POS:",pos, "; ROT:", rot)
        
        return new_frame
    else:
        print("NO ARUCO DETECTIONS")
        return frame

server = VideoServer(cap, port=8008)
server.create_route("/aruco_detection", "aruco_detection", processor)
server.start()

while input("input q to stop...") != "q":
    pass