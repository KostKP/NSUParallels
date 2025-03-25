import argparse
import logging
import os
import time
import queue
import cv2
from threading import Thread, Event

# Настройка логирования
if not os.path.exists('log'):
    os.makedirs('log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/application.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Базовый класс и датчики
class Sensor:
    def get(self):
        raise NotImplementedError()

class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0
    
    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, cam_name: int, resolution: tuple):
        self.cam_name = cam_name
        self.resolution = resolution
        self.cap = cv2.VideoCapture(self.cam_name)
        if not self.cap.isOpened():
            logger.error(f"Camera with index {cam_name} not found")
            raise RuntimeError("Camera initialization failed")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        logger.info(f"Camera initialized with resolution {resolution}")

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            raise RuntimeError("Camera read error")
        return frame

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released")

class WindowImage:
    def __init__(self, display_freq: float):
        self.display_freq = display_freq
        self.window_name = "Sensor Display"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def show(self, img):
        cv2.imshow(self.window_name, img)
    
    def __del__(self):
        cv2.destroyAllWindows()
        logger.info("Window destroyed")

def sensor_worker(sensor, data_queue, running, error_queue):
    while running.is_set():
        try:
            data = sensor.get()
            data_queue.put(data)
        except Exception as e:
            error_queue.put(str(e))
            logger.error(f"Sensor error: {e}")
            running.clear()
            break

def parse_resolution(res_str):
    try:
        w, h = map(int, res_str.split('x'))
        return (w, h)
    except:
        raise argparse.ArgumentTypeError("Invalid resolution format")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, required=True)
    parser.add_argument('--resolution', type=parse_resolution, required=True)
    parser.add_argument('--freq', type=float, required=True)
    args = parser.parse_args()

    sensors = [
        SensorX(0.01),
        SensorX(0.1),
        SensorX(1),
        SensorCam(args.camera, args.resolution)
    ]

    queues = [queue.Queue() for _ in range(4)]
    running = Event()
    running.set()
    error_queue = queue.Queue()

    threads = []
    for i, sensor in enumerate(sensors):
        t = Thread(
            target=sensor_worker,
            args=(sensor, queues[i], running, error_queue),
            daemon=True
        )
        t.start()
        threads.append(t)

    window = WindowImage(args.freq)
    display_interval = 1.0 / args.freq
    last_data = [0, 0, 0]
    last_frame = None

    try:
        next_display = time.time()
        while running.is_set():
            if not error_queue.empty():
                logger.error(f"Error: {error_queue.get()}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting...")
                break

            current_time = time.time()
            if current_time < next_display:
                time.sleep(max(0, next_display - current_time - 0.001))
                continue
            next_display += display_interval

            # Обновление кадра
            try:
                last_frame = queues[3].get_nowait()
            except queue.Empty:
                pass

            # Обновление данных датчиков
            for i in range(3):
                try:
                    while True:
                        last_data[i] = queues[i].get_nowait()
                except queue.Empty:
                    pass

            if last_frame is not None:
                h, w = last_frame.shape[:2]
                cv2.putText(last_frame, 
                    f"Sensor0: {last_data[0]} Sensor1: {last_data[1]} Sensor2: {last_data[2]}",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                window.show(last_frame)

    finally:
        running.clear()
        for t in threads:
            t.join(timeout=1)
        del window
        logger.info("Application stopped")

if __name__ == "__main__":
    main()