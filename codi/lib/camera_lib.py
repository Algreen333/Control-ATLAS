import time
import cv2
from flask import Flask, Response, redirect, request
from threading import Thread, Lock
import numpy as np
import json


try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except (ImportError, RuntimeError):
    HAS_PICAMERA = False


class VideoCapture:
    def __init__(self, capture_source = 0, resolution=(640, 480), format="RGB888", fps=30.0):
        """
        :param any capture_source: Source of the capture. Irrelevant when using Picamera2
        :param (int,int) resolution: Resolution of the capture. By default (640,480)
        :param str format: Format of the capture. Only relevant when using Picamera2. By default "RGB888"
        :param float fps: Fps of the capture. By default 30.0
        """
        
        self.resolution = resolution
        self.fps = fps
        self.format = format
        self.capture_source = capture_source
        
        if HAS_PICAMERA:
            self.capture = Picamera2(capture_source)
            config = self.capture.create_preview_configuration(main={"size": self.resolution, "format": self.format})
            self.capture.configure(config)

        else:
            self.capture = cv2.VideoCapture(self.capture_source)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        # For recording
        self.is_recording = False
        self.video_writer = None    

        self.frame = None
        self.stopped = False
        self.lock = Lock()
        
        self.capture_thread = Thread(target=self._update, daemon=True)

    def start(self):
        """
        Inicializes capture.

        :return self (VideoCapture):
        """


        if HAS_PICAMERA: self.capture.start()
        self.capture_thread.start()

        time.sleep(1) 

        print("[*] Video capture started")
        return self

    def _update(self):
        while not self.stopped:
            if HAS_PICAMERA:
                new_frame = self.capture.capture_array()
            else:
                ret, new_frame = self.capture.read()
                if not ret:
                    continue
                
            with self.lock:
                self.frame = new_frame

    def read(self):
        """
        Read a frame.
        
        :return ret (bool): Whether the frame has been read successfully
        :return frame (MatLike): The captured frame 
        """
        try:
            with self.lock:
                return True, self.frame
        except: 
            return False, None

    def toggle_record(self, filename=None):
        """
        Toggles record state. If it is not recording it creates opens a new VideoWriter and starts writing a video.
        
        :param str (optional) filename: Filename of the video. If left on None the video will be created with name "rec_time.avi"
        """
        if not self.is_recording:
            self.is_recording = True
            filename = filename | f"rec_{int(time.time())}.avi"
            self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.resolution)
            Thread(target=self._record_loop, daemon=True).start()
        else:
            self.is_recording = False

    def _record_loop(self):
        while self.is_recording:
            ret, f = self.read()
            if ret:    
                if f is not None: self.video_writer.write(f)
                time.sleep(1/self.fps)

        self.video_writer.release()

    def stop(self):
        """
        Stops video capture
        """

        self.stopped = True
        if self.capture_thread.is_alive():
            self.capture_thread.join()

        if HAS_PICAMERA:
            self.capture.stop()
            self.capture.close()
        else: 
            self.capture.release()


class VideoServer:
    def __init__(self, streamer:VideoCapture, host='0.0.0.0', port=5000):
        """
        Starts video server for a video capture. It is a Flask server which shows the capture in real time.

        :param VideoCapture streamer: VideoCapture whcih to show
        :param str host: Host address. By default "0.0.0.0"
        :param int port: Host port. By default 5000
        """

        self.streamer:VideoCapture = streamer
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        self.video_writer = None
        self.is_recording = False

        self.custom_endpoints = {"/video_feed": "Raw Stream"}
        
        # Register routes
        self.app.add_url_rule('/', 'index', lambda: self._render_page("/video_feed_raw"))
        self.app.add_url_rule('/video_feed', 'video_feed', lambda: self._render_page("/video_feed_raw"))
        self.app.add_url_rule('/video_feed_raw', 'video_feed_raw', self.video_feed_raw)
        self.app.add_url_rule('/toggle_record', 'toggle_record', self.toggle_record, methods=['POST'])

    def _render_page(self, stream_url):
        """The Master Template: Fullscreen with Hover UI"""
        rec_color = "#ff4444" if self.streamer.is_recording else "#44bb44"
        rec_text = "Stop Recording" if self.streamer.is_recording else "Start Recording"
        
        # Build navigation links for the overlay
        nav_links = "".join([f'<a href="{url}" style="color:white; margin-right:15px; text-decoration:none; font-size:14px; opacity:0.7;">{name}</a>' 
                             for url, name in self.custom_endpoints.items()])

        return f"""
        <html>
            <head>
                <style>
                    body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: black; font-family: sans-serif; }}
                    .viewport {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }}
                    .overlay {{ position: absolute; top: 0; left: 0; width: 100%; padding: 20px; box-sizing: border-box;
                                background: linear-gradient(to bottom, rgba(0,0,0,0.8) 0%, transparent 100%);
                                color: white; opacity: 0; transition: opacity 0.3s; z-index: 10; display: flex; justify-content: space-between; }}
                    .overlay:hover {{ opacity: 1; }}
                    .btn {{ padding: 10px 20px; background: {rec_color}; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }}
                </style>
            </head>
            <body>
                <img class="viewport" src="{stream_url}">
                <div class="overlay">
                    <div>
                        <h2 style="margin:0 0 10px 0;">PiStream Portal</h2>
                        {nav_links}
                    </div>
                    <form action="/toggle_record" method="post" style="margin:0;">
                        <input type="hidden" name="next" value="{stream_url}">
                        <button class="btn">{rec_text}</button>
                    </form>
                </div>
            </body>
        </html>
        """

    def toggle_record(self, filename = None):
        """
        Toggles record state. If it is not recording it creates opens a new VideoWriter and starts writing a video.
        
        :param str (optional) filename: Filename of the video. If left on None the video will be created with name "rec_time.avi"
        """

        if not self.is_recording:
            self.is_recording = True
            filename = f"recordings/rec_{int(time.time())}.avi"
            self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.streamer.fps, self.streamer.resolution)
        else:
            self.is_recording = False
            self.video_writer.release()
        return redirect(request.form.get('next', '/'))

    def _generate_frames(self, processor=None):
        prev_frame = None
        start_time = time.time()

        while True:
            try:
                ret, frame = self.streamer.read()

                if not ret or frame is None:
                    # Si hi ha un error o el frame no és vàlid.
                    # Si està gravant escriurà un altre cop el frame anterior
                    if self.is_recording and self.video_writer is not None:
                        time.sleep(max(0, (time.time() - start_time - (1/self.streamer.fps))))
                        if prev_frame is not None: self.video_writer.write(prev_frame)
                        start_time = time.time()
                    continue
                
                if processor is not None: frame = processor(frame)


                # Convert NumPy array to JPEG
                _, buffer = cv2.imencode('.jpg', frame)

                if self.is_recording and self.video_writer is not None:
                    time.sleep(max(0, (time.time() - start_time - (1/self.streamer.fps))))
                    self.video_writer.write(prev_frame)
                    start_time = time.time()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print("'_generate_frames' exception:", e)

                if prev_frame is not None:
                    yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def index(self):
        return Response(self._generate_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def start(self):
        server_thread = Thread(
            target=lambda: self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False),
            daemon=True
        )
        server_thread.start()
        print(f"[*] Web server live at http://{self.host}:{self.port}")

    def create_route(self, rule, endpoint_name, generator):
        """
        Exemple: create_route('/aruco_detector', 'aruco_detector', aruco_detector), 
        on aruco_detector() és una funció aruco_detector(frame) -> frame_processat
        """
        self.custom_endpoints[rule] = endpoint_name
        self.app.add_url_rule(f"{rule}_raw", f"{endpoint_name}_raw", 
                              lambda: Response(self._generate_frames(processor=generator), 
                                    mimetype='multipart/x-mixed-replace; boundary=frame')
        )
        self.app.add_url_rule(rule, endpoint_name, 
                              lambda: self._render_page(f"{rule}_raw"))
    
    def video_feed_raw(self):
        return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


class CalibrationConfig:
    """
    Can be initialized with:
    - CalibrationConfig(mtx:np.ndarray, dist:np.ndarray) 
    - CalibrationConfig.from_path(path): path is the config file path
    - CalibrationConfig.from_lists(mtx:list, dist:list)
    """

    def __init__(self, mtx: np.ndarray, dist: np.ndarray):
        # The main constructor only accepts the finalized numpy arrays
        self.mtx = mtx
        self.dist = dist

    @classmethod
    def from_path(self, path: str):
        # Read the file, extract matrices
        with open(path, "r") as f:
            data = json.load(f)

        loaded_mtx = np.array(data["mtx"])
        loaded_dist = np.array(data["dist"])

        return self(loaded_mtx, loaded_dist)

    @classmethod
    def from_lists(self, mtx: list, dist: list):
        return self(np.array(mtx), np.array(dist))
    
    def save(self, path:str):
        config_data = {
            "mtx": self.mtx.tolist(),
            "dist": self.dist.tolist()
        }
        with open(path, "w") as f:
            json.dump(config_data, f, indent=4)
                

if __name__ == "__main__":

    cap = VideoCapture(0)
    cap.start()

    server = VideoServer(cap)
    server.start()