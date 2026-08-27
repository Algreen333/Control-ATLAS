from lib.drone_lib import *
from lib.aruco_lib import *
from lib.camera_lib import *
from lib.utils import *

import math

ALTURA_VOL = 3              # Utilitzada al takeoff
THRESH_DIST = 0.25          # Dist màxima per poder aterrar
AVG_POS_LEN = 6             # Quantes mostres de la posició de l'aruco es prenen
MAX_MOVE_REL = 1            # Distància màxima que es pot moure en una sola acció
SCALE_FACTOR = 0.9          # Escalat de la distància de moviment respecte la calculada
DOWN_STEP = 0.2             # Distància que descendeix a cada moviment que fa el dron cap a l'aruco
DELAY_BETWEEN_MOVES = 4     # Temps que s'espera després d'executar una comanda de Mavlink

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
        # Espera a tenir un input "y" abans de fer el takeoff
        while input("start? (y)") != "y": print("Not starting")
        print("TRIGGER_INICI")

        # Sequencia takeoff
        con.takeoff(ALTURA_VOL)
        time.sleep(10)
        #if not con.armat_i_guided(): 
        #    raise "S'HA DESARMAT O S'HA PRES CONTROL MANUAL"
        
        # GOTOARUCO
        print("INICIANT GOTOARUCO...")

        poss = []

        dist = 2*THRESH_DIST

        while dist > THRESH_DIST:
            det = None
            while det is None: 
                ret, frame = cap.read()
                cv2.rotate(frame, 180)
                if ret: det = aruco_detector.full_prediction(frame)
                else: last_frame = frame

            last_frame, pos, rot = det
            
            poss.append(pos)
            if len(poss) >= AVG_POS_LEN:
                # PERFORM MOVE ACTION
                x,y,z = average_pos(poss)
                dist = math.sqrt(x*x + y*y)

                x *= SCALE_FACTOR
                y *= SCALE_FACTOR

                # Clamp vals

                x = clamp(x, -MAX_MOVE_REL, MAX_MOVE_REL)
                y = clamp(y, -MAX_MOVE_REL, MAX_MOVE_REL)
                
                print("MOVENT... x:",x,"; y:", y, "dist:", dist)
                con.move_relative(y, -x, 0.2)
                poss.clear()

                start_t = time.time()
                while (time.time() < start_t + DELAY_BETWEEN_MOVES):
                    ret, frame = cap.read()
                    cv2.rotate(frame, 180)
                    if ret: 
                        det = aruco_detector.full_prediction(frame)
                        last_frame, pos, rot = det
                    else: last_frame = frame
        
        # QUAN APROP
        print("ATERRANT...")
        con.land()

        time.sleep(5)
        print("ATERRAT, ESPERANT RESET MANUAL")

        con.esperar_reset_manual()