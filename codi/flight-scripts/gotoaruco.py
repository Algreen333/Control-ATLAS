from lib.aruco_lib import *
from lib.drone_lib import *
from lib.camera_lib import *

ALTURA_VOL = 3
THRESH_DIST = 0.2
AVG_POS_LEN = 6
MAX_MOVE_REL = 1

last_frame = None

def processor(frame):
    return last_frame


if __name__ == "__main__":
    # CAPTURE
    cap = VideoCapture()
    cap.start()

    ret = False
    while not ret:
        ret, last_frame = cap.read()

    # CALIBRATION
    mtx = [
        [665.15857148, 0,              319.15873703],
        [0,            662.49958721,   264.6089531 ],
        [0,            0,              1]
    ]
    dist = [[ 0.19320481, 0.59536031,0.01756988,-0.01678634,-2.19110878]]
    aruco_detector = ArucoDetector(mtx, dist)

    server = VideoServer(cap)
    server.create_route("/aruco_detection", "aruco_detection", processor)
    server.start()

    con = MavlinkConnection()

    while True:
        con.esperar_trigger_inici()
        
        # Sequencia takeoff
        con.takeoff(ALTURA_VOL)
        time.sleep(10)
        if not con.armat_i_guided(): 
            raise "S'HA DESARMAT O S'HA PRES CONTROL MANUAL"
        
        # GOTOARUCO
        print("INICIANT GOTOARUCO...")

        poss = []

        dist = 2*THRESH_DIST

        while dist > THRESH_DIST:
            det = None
            while det is None: 
                ret, frame = cap.read()
                if ret: det = aruco_detector.full_prediction(frame)

            last_frame, pos, rot = det
            
            poss.append(pos)
            if len(poss) >= AVG_POS_LEN:
                # PERFORM MOVE ACTION
                x,y,z = average_pos(poss)
                dist = np.linalg.norm((x,y,z))

                # Clamp vals
                if (abs(x/dist*MAX_MOVE_REL) < abs(x)): x = x/dist * MAX_MOVE_REL
                if (abs(y/dist*MAX_MOVE_REL) < abs(y)): y = y/dist * MAX_MOVE_REL
                print("MOVENT...")
                con.move_relative(-y, x, 0)
                time.sleep(2)
        
        # QUAN APROP
        print("ATERRANT...")
        con.land()

        time.sleep(5)
        print("ATERRAT, ESPERANT RESET MANUAL")

        con.esperar_reset_manual()