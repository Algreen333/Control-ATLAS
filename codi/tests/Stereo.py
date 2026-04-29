import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (main)
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lib.camera_lib import *
import cv2
from picamera2 import Picamera2
from libcamera import controls
import time

FPS = 41.0
#ctrlsn = {'FrameRate': FPS, 'SyncMode': controls.rpi.SyncModeEnum.Server}
#ctrlsw = {'FrameRate': FPS, 'SyncMode': controls.rpi.SyncModeEnum.Client}
ctrlsn = {"FrameRate": FPS}
ctrlsw = {"FrameRate": FPS}

widecap = Picamera2(0)
narrowcap = Picamera2(1)

# Primer inicialitzem el client
confw = widecap.create_preview_configuration(main={"size": (1536, 864)}, controls=ctrlsw)
# Després el server
confn = narrowcap.create_preview_configuration(main={"size": (1640, 1232)}, controls=ctrlsn)

wideconf = CalibrationConfig.from_path("../configs/1536x864-v3.conf")
narrowconf = CalibrationConfig.from_path("../configs/1640x1232-v2.conf")

widecap.start(confw)
narrowcap.start(confn)

#req = widecap.capture_sync_request()
time.sleep(1)
#print("Sync error:", req.get_metadata()['SyncTimer'])

while True:
    
    y = narrowcap.capture_request()
    x = widecap.capture_request()
    
    m1 = x.get_metadata()
    m2 = y.get_metadata()

    print(m1, "\n", m2, "\n", abs(m1["SensorTimestamp"] - m2["SensorTimestamp"]))
    
    time.sleep(1)
