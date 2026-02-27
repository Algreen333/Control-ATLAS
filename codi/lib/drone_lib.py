from pymavlink import mavutil

# Coses a mirar:
# https://mavlink.io/en/messages/common.html#MAV_CMD_REQUEST_MESSAGE -> request_message
# https://mavlink.io/en/messages/common.html#COMMAND_INT -> COMMAND_INT
# https://mavlink.io/en/messages/common.html#ORBIT_YAW_BEHAVIOUR -> Behaviour 
# https://mavlink.io/en/messages/common.html#MAV_FRAME -> Reference frame




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