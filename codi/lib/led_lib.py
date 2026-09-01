import time
import threading
import logging

logger = logging.getLogger(__name__)

# Dynamically check for Pi5Neo
try:
    from pi5neo import Pi5Neo
    HAS_PI5NEO = True
except ImportError:
    HAS_PI5NEO = False
    logger.info("[LED] 'pi5neo' not installed. LED changes will be logged to console.")

# --- LED Configuration ---
NUM_LEDS  = 16                 
SPI_DEV   = '/dev/spidev0.0'  
SPI_KHZ   = 800               
BRIGHTNESS = 0.3              

COLOR_VERD    = (0, 255, 0)
COLOR_VERMELL = (255, 0, 0)
COLOR_BLAU    = (0, 0, 255)
COLOR_GROC    = (255, 180, 0)
COLOR_CIAN    = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_APAGAT  = (0, 0, 0)

MODE_COLORS = {
    'GUIDED':    COLOR_VERD,      
    'AUTO':      COLOR_VERD,      
    'STABILIZE': COLOR_VERMELL,   
    'ALT_HOLD':  COLOR_VERMELL,   
    'LOITER':    COLOR_CIAN,      
    'RTL':       COLOR_GROC,      
    'LAND':      COLOR_GROC,      
    'POSHOLD':   COLOR_CIAN,
}
COLOR_PER_DEFECTE = COLOR_BLAU 

class LEDController:
    def __init__(self):
        self.current_mode = None
        self.last_applied_mode = None
        self.last_heartbeat_time = 0
        
        if HAS_PI5NEO:
            self.neo = Pi5Neo('/dev/spidev0.0', 16, 800)
            
        self._fill_leds((0, 0, 255), "BOOT (BLAU)") 

    def start(self):
        threading.Thread(target=self._led_loop, daemon=True).start()

    def update_state(self, flightmode: str):
        """Called instantly by the main message pump when a heartbeat arrives."""
        if self.current_mode != flightmode:
            self.current_mode = flightmode
            if flightmode in MODE_COLORS: self._fill_leds(MODE_COLORS[flightmode], flightmode)
            else: logger.warning(f"UNKNOWN FLIGHT MODE {flightmode}")
        self.last_heartbeat_time = time.monotonic()

    def _fill_leds(self, color, label=""):
        if HAS_PI5NEO:
            r, g, b = (int(c * 0.3) for c in color)
            self.neo.fill_strip(r, g, b)
            self.neo.update_strip()
        else:
            logger.debug(f"[LED SIMULATION] Changed to {color} {label}")

    def _led_loop(self):
        while True:
            # Flash magenta on link loss
            if time.monotonic() - self.last_heartbeat_time > 3.0 and self.last_heartbeat_time != 0:
                if HAS_PI5NEO:
                    self._fill_leds((255, 0, 255))
                    time.sleep(0.25)
                    self._fill_leds((0, 0, 0))
                    time.sleep(0.25)
                else:
                    logger.debug("[LED SIMULATION] Flashing MAGENTA (Link Lost)")
                    time.sleep(0.5)
                self.last_applied_mode = None
                continue

            # Update normal flight modes
            if self.current_mode != self.last_applied_mode:
                color = MODE_COLORS.get(self.current_mode, (0, 0, 255))
                self._fill_leds(color, f"(Mode: {self.current_mode})")
                self.last_applied_mode = self.current_mode
            
            time.sleep(0.05)