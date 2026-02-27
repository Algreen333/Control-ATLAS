### EXECUTAR ESTANT A "main"
import sys
sys.path.append("../")

from lib.aruco_lib import *
from lib.camera_lib import *

cap = VideoCapture()
cap.start()


mtx = [
    [665.15857148, 0.00000000e+00, 319.15873703],
    [0.00000000e+00, 662.49958721, 264.6089531],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
]

dist = [
    [0.19320481, 0.59536031, 0.01756988, -0.01678634, -2.19110878]
]

aruco_detector = ArucoDetector(mtx, dist)

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