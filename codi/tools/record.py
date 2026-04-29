import time
import subprocess
import os
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from threading import Thread, Event

from datetime import datetime
example = "2026-04-24T16:40:19"
parsed = datetime.strftime(datetime.now(), "%Y-%m-%d--%H-%M-%S")


def record_camera(camera_id, raw_file, resolution, stop_event):
    """Records raw H.264 video from a specific camera."""
    pic = None
    try:
        pic = Picamera2(camera_id)
        config = pic.create_video_configuration(main={"size": resolution})
        pic.configure(config)
        pic.start()
        
        print(f"Cam {camera_id} initialized: {resolution[0]}x{resolution[1]}")
        
        # Using the standard two-argument syntax for raw H.264
        encoder = H264Encoder()
        output = FileOutput(raw_file)
        pic.start_recording(encoder, output)
        
        while not stop_event.is_set():
            time.sleep(0.1)
            
        pic.stop_recording()
        print(f"Cam {camera_id} raw data saved.")
        
    except Exception as e:
        print(f"Error on Camera {camera_id}: {e}")
    finally:
        if pic:
            pic.stop()
            pic.close()

def convert_to_mp4(input_file, output_file):
    """Wraps raw H.264 into an MP4 container using FFmpeg."""
    print(f"Converting {input_file} to {output_file}...")
    try:
        # '-c copy' is instantaneous because it doesn't re-encode the video
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file, 
            '-c', 'copy', output_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # Clean up the raw file after successful conversion
        os.remove(input_file)
        print(f"Finished: {output_file}")
    except Exception as e:
        print(f"Conversion error for {input_file}: {e}")

if __name__ == "__main__":
    # --- SETTINGS ---
    # We record to .h264 first, then convert to .mp4

    

    RAW_0, FINAL_0 = "temp_cam0.h264", f"{parsed}-wide.mp4"
    RAW_1, FINAL_1 = "temp_cam1.h264", f"{parsed}-narrow.mp4"
    
    RES_0 = (2304, 1296)
    RES_1 = (1640, 1232)
    # ----------------

    stop_signal = Event()

    # Step 1: Start Recording
    t0 = Thread(target=record_camera, args=(0, RAW_0, RES_0, stop_signal))
    t1 = Thread(target=record_camera, args=(1, RAW_1, RES_1, stop_signal))

    t0.start()
    t1.start()

    print("\n--- RECORDING ---")
    print("Type 'stop' and press Enter to finish.")
    
    while True:
        if input().strip().lower() == "stop":
            stop_signal.set()
            break

    t0.join()
    t1.join()

    print("\n--- POST-PROCESSING ---")
    # Step 2: Convert to MP4
    convert_to_mp4(RAW_0, FINAL_0)
    convert_to_mp4(RAW_1, FINAL_1)

    print("\nAll tasks complete. Files are ready.")