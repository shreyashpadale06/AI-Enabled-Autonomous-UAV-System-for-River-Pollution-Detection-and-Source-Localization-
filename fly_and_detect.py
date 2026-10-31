"""On-board pipeline: camera -> YOLOv8n -> geotag -> log -> hotspots.
Modes:
  --source picam            Raspberry Pi camera + Pixhawk GPS over MAVLink
  --source video.mp4 --gps track.csv   offline test (csv: t,lat,lon)
"""
import argparse, csv, json, time, os
import cv2
from ultralytics import YOLO
from hotspot import Detection, find_hotspots

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/river_yolov8n/weights/best.pt")
ap.add_argument("--source", default="picam")
ap.add_argument("--gps", default=None)
ap.add_argument("--mav", default="/dev/serial0,921600")   # Pixhawk TELEM port
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--out", default="mission")
a = ap.parse_args()
os.makedirs(f"{a.out}/frames", exist_ok=True)
model = YOLO(a.weights)

# ---- position source
if a.source == "picam":
    from pymavlink import mavutil
    from picamera2 import Picamera2
    dev, baud = a.mav.split(",")
    mav = mavutil.mavlink_connection(dev, baud=int(baud)); mav.wait_heartbeat()
    cam = Picamera2(); cam.configure(cam.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})); cam.start()
    def next_frame(): return cam.capture_array()
    def position(_):
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        return (msg.lat/1e7, msg.lon/1e7) if msg else None
else:
    cap = cv2.VideoCapture(a.source); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    track = [tuple(map(float, r)) for r in list(csv.reader(open(a.gps)))[1:]]
    def next_frame():
        ok, f = cap.read(); return f if ok else None
    def position(i):
        t = i / fps
        return min(track, key=lambda p: abs(p[0]-t))[1:]

dets, i = [], 0
log = open(f"{a.out}/detections.jsonl", "w")
try:
    while True:
        frame = next_frame()
        if frame is None: break
        pos = position(i); i += 1
        if pos is None: continue
        res = model.predict(frame, imgsz=640, conf=a.conf, verbose=False)[0]
        for b in res.boxes:
            path = f"{a.out}/frames/{i:06d}.jpg"
            if not os.path.exists(path): cv2.imwrite(path, res.plot())
            d = Detection(model.names[int(b.cls)], float(b.conf), pos[0], pos[1], i, time.time(), path)
            dets.append(d); log.write(json.dumps(d.__dict__) + "\n"); log.flush()
except KeyboardInterrupt:
    pass
hs = find_hotspots(dets)
json.dump([h.__dict__ for h in hs], open(f"{a.out}/hotspots.json", "w"), indent=2)
print(f"{len(dets)} detections -> {len(hs)} hotspots (see {a.out}/hotspots.json)")
