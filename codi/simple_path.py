from lib.drone_lib import *

import time

# =========================================================================
#  simple_path.py
#  Takeoff a ALTURA_VOL, recorregut en forma de quadrat (centre = punt de
#  takeoff) i tornada al punt de takeoff per aterrar.
#  El moviment es fa amb coordenades GLOBALS (GPS): veure drone_lib.goto_global.
#
#  PRECONDICIONS (igual que gotoaruco):
#    - El dron ha d'estar ARMAT i en mode GUIDED abans d'escriure "y".
#    - Ha de tenir fix de GPS (obligatori per armar en GUIDED a ArduPilot).
# =========================================================================

# ---- Paràmetres de la missió ----
ALTURA_VOL = 2.0            # Alçada de takeoff i de tot el recorregut (m)
COSTAT = 2.0               # Costat del quadrat (m)
REACH_THRESH = 0.4         # Radi (m) per considerar un waypoint assolit
WAYPOINT_TIMEOUT = 20      # Temps màxim (s) esperant a assolir un waypoint
RESEND_PERIOD = 1.0        # Cada quant es reenvia el setpoint mentre s'espera (s)
SETTLE_AFTER_TAKEOFF = 10  # Estabilització després del takeoff (s)


def fly_to(con, lat, lon, alt, label=""):
    """
    Vola cap a (lat, lon, alt) i espera fins a assolir-lo (dins REACH_THRESH)
    o fins a WAYPOINT_TIMEOUT. Reenvia el setpoint periòdicament per robustesa.
    """
    print(f"-> Anant a {label}  lat={lat:.7f} lon={lon:.7f}")
    con.goto_global(lat, lon, alt)

    start = time.time()
    last_send = start
    while True:
        pos = con.get_global_position(timeout=2)
        if pos is not None:
            c_lat, c_lon, c_alt = pos
            d = horizontal_distance(c_lat, c_lon, lat, lon)
            print(f"   dist={d:.2f} m  alt={c_alt:.2f} m")
            if d <= REACH_THRESH:
                print(f"   [OK] {label} assolit")
                return

        now = time.time()
        if now - last_send >= RESEND_PERIOD:
            con.goto_global(lat, lon, alt)   # reenviar per si es perd un paquet
            last_send = now
        if now - start > WAYPOINT_TIMEOUT:
            print(f"   [TIMEOUT] Continuant sense haver assolit {label}")
            return
        time.sleep(0.1)


if __name__ == "__main__":
    con = MavlinkConnection()
    con.request_global_position_stream(hz=5)

    while True:
        # Espera un input "y" abans de fer el takeoff
        while input("start? (y)") != "y":
            print("Not starting")
        print("TRIGGER_INICI")

        # Sequencia takeoff
        con.takeoff(ALTURA_VOL)
        time.sleep(SETTLE_AFTER_TAKEOFF)

        # SIMPLE PATH
        print("INICIANT SIMPLE_PATH...")

        # Centre del quadrat = punt de takeoff (posició actual estabilitzada)
        center = None
        while center is None:
            center = con.get_global_position(timeout=2)
            if center is None:
                print("Esperant GPS (GLOBAL_POSITION_INT)...")
        c_lat, c_lon, _ = center
        print(f"CENTRE (takeoff): lat={c_lat:.7f} lon={c_lon:.7f}")

        # Recorregut: centre -> C1 -> C2 -> C3 -> C4 -> C1 (tanca) -> centre
        h = COSTAT / 2.0
        corners = [
            ("C1 (NE)",  h,  h),
            ("C2 (NO)",  h, -h),
            ("C3 (SO)", -h, -h),
            ("C4 (SE)", -h,  h),
            ("C1 (NE)",  h,  h),   # torna a la primera cantonada per tancar el quadrat
        ]

        for label, d_north, d_east in corners:
            w_lat, w_lon = offset_location(c_lat, c_lon, d_north, d_east)
            fly_to(con, w_lat, w_lon, ALTURA_VOL, label)

        # Tornar al punt de takeoff
        fly_to(con, c_lat, c_lon, ALTURA_VOL, "CENTRE (takeoff)")

        # QUAN APROP
        print("ATERRANT...")
        con.land()

        time.sleep(5)
        print("ATERRAT, missio acabada.")