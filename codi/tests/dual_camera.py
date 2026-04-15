import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (main)
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lib.camera_lib import *
from lib.aruco_lib import *
import cv2

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

WIDE_SZ = (1536,864)
NARROW_SZ = (1640,1232)

widecap = VideoCapture(0, resolution=WIDE_SZ)
wideconf = CalibrationConfig.from_path("../configs/1536x864-v3.conf")

narrowcap = VideoCapture(1, resolution=NARROW_SZ)
narrowconf = CalibrationConfig.from_path("../configs/1640x1232-v2.conf")

widedet = ArucoDetector(wideconf.mtx, wideconf.dist)
narrowdet = ArucoDetector(narrowconf.mtx, narrowconf.dist)


def resize_with_ratio(img, w_target):
        h_orig, w_orig = img.shape[:2]
        # Calculate aspect ratio
        aspect_ratio = h_orig / w_orig
        h_target = int(w_target * aspect_ratio)
        # Resize using inter_area for better downscaling quality
        return cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_AREA)

def dual_image_display(frame1, frame2):
    wide_res = resize_with_ratio(frame1, int(DISPLAY_WIDTH/2))
    narrow_res = resize_with_ratio(frame2, int(DISPLAY_WIDTH/2))

    h1, w1 = wide_res.shape[:2]
    h2, w2 = narrow_res.shape[:2]

    
    canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

    # Centra imatges
    y_off1 = int((DISPLAY_HEIGHT - h1) / 2)
    y_off2 = int((DISPLAY_HEIGHT - h2) / 2)

    canvas[y_off1 : y_off1 + h1, 0:w1] = wide_res
    canvas[y_off2 : y_off2 + h2, int(DISPLAY_WIDTH/2) : int(DISPLAY_WIDTH/2 + w2)] = narrow_res
    
    return canvas

def processor2(frame):
    try:
        ret, frame2 = narrowcap.read()
        if ret and frame2 is not None:
            canvas = dual_image_display(frame, frame2)
            return canvas
        
        return frame
    
    except Exception as e:
        print("'processor2' exception:", e)
        return frame

def processor(frame):
    try:
        ret, frame2 = narrowcap.read()
        if ret and frame2 is not None:
            retwide = widedet.full_prediction(frame)
            retnarrow = narrowdet.full_prediction(frame2)
            
            if retwide is not None:
                x_off, y_off, z_off = retwide[1]
                pitch, yaw, roll = retwide[2]
                print("----------------- WIDE CAM -----------------")
                print(f"Position: x={x_off:.2f}, y={y_off:.2f}, z={z_off:.2f}")
                print(f"Orientation: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}")
                print("--------------------------------------------")
                frame = retwide[0]

            if retnarrow is not None:
                x_off, y_off, z_off = retnarrow[1]
                pitch, yaw, roll = retnarrow[2]
                print("---------------- NARROW CAM ----------------")
                print(f"Position: x={x_off:.2f}, y={y_off:.2f}, z={z_off:.2f}")
                print(f"Orientation: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}")
                print("--------------------------------------------")
                frame2 = retnarrow[0]

            canvas = dual_image_display(frame, frame2)
            time.sleep(0.033)

            return canvas
        
        return frame
    
    except Exception as e:
        print("'processor' exception:", e)
        return frame

if __name__ == "__main__":

    widecap.start()
    narrowcap.start()

    server = VideoServer(widecap, port=8008)
    server.create_route("/aruco_detection", "aruco_detection", processor)
    server.create_route("/dual", "dual", processor2)
    server.start()

    while input("input q to stop...") != "q":
        pass