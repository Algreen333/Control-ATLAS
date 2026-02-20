### EXECUTAR ESTANT A "main"

from lib.aruco_lib import *
from lib.camera_lib import *

cap = VideoCapture()
cap.start()


mtx = [
    [1.47994555e+03, 0.00000000e+00, 9.62446789e+02],
    [0.00000000e+00, 1.48121216e+03, 5.09517452e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
]

dist = [
    [-0.00607252, 0.31175609, -0.00779087, 0.00107362, -0.64724446]
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