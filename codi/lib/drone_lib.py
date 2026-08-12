from pymavlink import mavutil
import math

# Coses a mirar:
# https://mavlink.io/en/messages/common.html#MAV_CMD_REQUEST_MESSAGE -> request_message
# https://mavlink.io/en/messages/common.html#COMMAND_INT -> COMMAND_INT
# https://mavlink.io/en/messages/common.html#ORBIT_YAW_BEHAVIOUR -> Behaviour 
# https://mavlink.io/en/messages/common.html#MAV_FRAME -> Reference frame
# https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_GLOBAL_INT -> Global setpoint


EARTH_RADIUS = 6378137.0   # Radi equatorial WGS84 (m), per a conversions lat/lon <-> metres


class MavlinkConnection:

    def __init__(self, connection_string = '/dev/serial0', baud_rate = 57600):
        self.connection_string = connection_string
        self.baud_rate = baud_rate

        ### INICIALITZAR MAVLINK
        self.master = mavutil.mavlink_connection(connection_string, baud=baud_rate)
        self.master.wait_heartbeat()

        # Obtenim els IDs dels modes necessaris
        #self.mode_guided_id = self.master.mode_mapping()['GUIDED']
        #self.mode_land_id = self.master.mode_mapping()['LAND']

    def move_relative(self, vertical:int|float, horizontal:int|float, updown:int|float):
        """
        Moves the drone (vertical, horizontal, updown) meters relative to the position and direction of the drone.
         
        :param int|float vertical: Positive is forward
        :param int|float horizontal: Positive is right
        :param int|float updown: Positive is down
        """
        self.master.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            0, 
            self.master.target_system, 
            self.master.target_component, 

            mavutil.mavlink.MAV_FRAME_BODY_FRD, 

            0b010111111000, 
            vertical, 
            horizontal, 
            updown, 

            0, 0, 0, 0, 0, 0, 0, 0))

    # ------------------------------------------------------------------ #
    #  MOVIMENT AMB COORDENADES GLOBALS (GPS)                            #
    # ------------------------------------------------------------------ #

    def request_global_position_stream(self, hz:int|float=5):
        """
        Demana a l'autopilot que emeti GLOBAL_POSITION_INT a 'hz' Hz.
        Útil per garantir telemetria de posició fresca abans de moure's.

        :param int|float hz: Freqüència desitjada (Hz).
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,

            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,

            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            int(1e6 / hz),   # interval en microsegons
            0, 0, 0, 0, 0)

    def get_global_position(self, timeout:int|float=2):
        """
        Llegeix la posició global actual a partir de GLOBAL_POSITION_INT.

        :param int|float timeout: Temps màxim d'espera del missatge (s).
        :return (lat, lon, rel_alt) en (graus, graus, m), o None si no arriba res.
        """
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=timeout)
        if msg is None:
            return None
        return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0

    def goto_global(self, lat:float, lon:float, alt:float):
        """
        Envia un setpoint de posició global (només posició; velocitat,
        acceleració i yaw ignorats). L'autopilot vola cap al punt i s'hi manté.
        Requereix mode GUIDED i el dron armat.

        :param float lat: Latitud en graus decimals.
        :param float lon: Longitud en graus decimals.
        :param float alt: Alçada RELATIVA al punt d'armat (m).
        """
        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,

            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,

            0b110111111000,          # type_mask: només posició
            int(lat * 1e7),
            int(lon * 1e7),
            alt,

            0, 0, 0,                 # vx, vy, vz
            0, 0, 0,                 # afx, afy, afz
            0, 0)                    # yaw, yaw_rate

    def loiter_unlim(self, F, R, D):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            
            mavutil.mavlink.MAV_FRAME_BODY_FRD, # Reference frame

            0, 0,   # Unused

            0, 0,   # Empty
            None,   # Heading, None for NAN
            F,
            R,
            D
        )

    def orbit(self, radius:int|float, behaviour=3, rotations=0, lat=None, lon=None, height=None, reference=mavutil.mavlink.MAV_FRAME_LOCAL_NED):
        self.master.mav.send(
            self.master.target_system,
            self.master.target_component,
            reference,
            mavutil.mavlink.MAV_CMD_DO_ORBIT,
            
            0,
            0,
            radius,
            behaviour,
            rotations,
            lat,
            lon,
            height
            )
        
    def land(self):
        """
        Sends land command
        """

        self.master.mav.command_long_send(
            self.master.target_system, 
            self.master.target_component,

            mavutil.mavlink.MAV_CMD_NAV_LAND,

            0, 0, 0, 0, 0, 0, 0, 0)
    
    def takeoff(self, height:float=2):
        """
        Sends takeoff command.

        :param float height: Target takeoff altitude (in meters). 
        """
        self.master.mav.command_long_send(
        self.master.target_system, 
        self.master.target_component,
        
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,

        0, 0, 0, 0, 0, 0, 0, 
        height)

    def armat_i_guided(self):
        """
        NO FUNCIONAL DE MOMENT!!!!!!!!!!!
        """
        msg = self.master.recv_match(type='HEARTBEAT', blocking=True)
        if msg:
            esta_armat = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            es_guided = (msg.custom_mode == self.mode_guided_id)
            if esta_armat & es_guided: return True
            print("armat:", esta_armat)
            print("guided:", es_guided)
        return False
    
    def esperar_trigger_inici(self):
        """
        NO FUNCIONAL DE MOMENT!!!!!!!!!!!

        Bucle espera, comprovar abans de passar autònom:
            - Motors ARMATS
            - Mode GUIDED
        """
        print(">>> Esperant GUIDED o ARMAT...")
        
        while True:
            if self.armat_i_guided():
                print(">>> Dron preparat: Començant missió")
                # Buidem buffer per seguretat abans de començar
                while self.master.recv_match(blocking=False): pass
                return

    def esperar_reset_manual(self):
        """ 
        NO FUNCIONAL DE MOMENT!!!!!!!!!!!

        Espera que es tregui mode GUIDED
        """
        print("Treu mode GUIDED per poder seguir")
        
        while True:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True)
            if msg:
                if msg.custom_mode != self.mode_guided_id:
                    print(">>> Reset completat")
                    return

def offset_location(lat:float, lon:float, d_north:float, d_east:float):
    """
    Desplaça una posició lat/lon 'd_north' i 'd_east' metres (Nord/Est positius).
    Aproximació de terra plana, precisa de sobres a escala de pocs metres.

    :return (lat, lon) desplaçats, en graus decimals.
    """
    d_lat = (d_north / EARTH_RADIUS) * (180.0 / math.pi)
    d_lon = (d_east / (EARTH_RADIUS * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    return lat + d_lat, lon + d_lon


def horizontal_distance(lat1:float, lon1:float, lat2:float, lon2:float):
    """
    Distància horitzontal (m) entre dues posicions lat/lon (aprox. terra plana).
    """
    d_north = math.radians(lat2 - lat1) * EARTH_RADIUS
    d_east = math.radians(lon2 - lon1) * EARTH_RADIUS * math.cos(math.radians(lat1))
    return math.sqrt(d_north * d_north + d_east * d_east)