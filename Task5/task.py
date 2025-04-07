import argparse
import time
import cv2
import threading
from queue import Queue
from ultralytics import YOLO
import os

class ModelWrapper:
    def __init__(self):
        self.model = YOLO('yolov8s-pose.pt')
        self.model.to('cpu')  # Используем CPU
    def __del__(self):
        del self.model

    def infer(self, frame):
        return self.model.predict(frame, verbose=False)[0]

def single_thread_process(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    model = ModelWrapper()

    frame_count = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = model.infer(frame)
        annotated = result.plot(boxes=False, labels=False)
        out.write(annotated)
        frame_count += 1
    end_time = time.time()

    cap.release()
    out.release()

    print(f"[Single Thread] Processed {frame_count} frames in {end_time - start_time:.2f} seconds")

def multi_thread_process(video_path, output_path, num_workers):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w, frame_h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    input_queue = Queue()
    output_dict = {}
    lock = threading.Lock()
    finished_event = threading.Event()

    def worker(worker_id):
        model = ModelWrapper()
        while not finished_event.is_set():
            try:
                index, frame = input_queue.get(timeout=1)
            except:
                continue
            result = model.infer(frame)
            annotated = result.plot(boxes=False, labels=False)
            with lock:
                output_dict[index] = annotated
            input_queue.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(num_workers)]
    for t in threads:
        t.start()

    index = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_queue.put((index, frame))
        index += 1

    cap.release()
    input_queue.join()
    finished_event.set()

    # Собрать кадры в правильном порядке
    for i in range(index):
        while i not in output_dict:
            time.sleep(0.01)
        out.write(output_dict[i])
    out.release()

    end_time = time.time()
    print(f"[Multi Thread ({num_workers})] Processed {index} frames in {end_time - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation Video Processor")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (640x480)")
    parser.add_argument("--instances", type=int, default=1, help="Number of model instances (1 = single thread)")
    parser.add_argument("--output", type=str, required=True, help="Output video filename")

    args = parser.parse_args()

    if args.instances == 1:
        single_thread_process(args.video, args.output)
    else:
        multi_thread_process(args.video, args.output, args.instances)

if __name__ == "__main__":
    main()
