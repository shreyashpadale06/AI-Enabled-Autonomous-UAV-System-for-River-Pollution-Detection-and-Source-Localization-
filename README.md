# AI-Enabled-Autonomous-UAV-System-for-River-Pollution-Detection-and-Source-Localization-

The UAV River Pollution Detector uses the YOLOv8n model to detect wastewater, sewage outfalls, and floating waste in rivers. Images are collected and labelled using tools such as Roboflow, CVAT, or LabelImg, and the dataset is split into training and testing sets based on individual flights. The model is trained using `train.py` and evaluated using `evaluate.py`. The trained model (`best.pt`) is then deployed on a Raspberry Pi, with optional NCNN conversion to improve detection speed. The system supports real-time detection through a Pi camera or offline video analysis with GPS data. For better accuracy, the dataset should include at least 1,500 images per class under different environmental conditions, along with negative images to reduce false detections. Finally, detected pollution hotspots should be verified using water-quality sensors or laboratory tests, as vision-based detection alone cannot confirm actual water pollution.


UAV River Pollution Detector (YOLOv8n)
pip install -r requirements.txt
Collect/label images (Roboflow, CVAT, LabelImg) in YOLO format, classes: wastewater, sewage_outfall, floating_waste. Name files <flightid>_<n>.jpg + .txt, then python split_dataset.py raw dataset.
python train.py -> python evaluate.py
Copy best.pt to the Pi. Optional speed-up: yolo export model=best.pt format=ncnn imgsz=640, then pass the exported folder as --weights.
python fly_and_detect.py --source picam (or offline: --source clip.mp4 --gps track.csv)
Getting genuinely good results
1,500+ labelled images per class, from different rivers, light, seasons, altitudes (10-30 m).
Split by flight, never by random frame (split_dataset.py does this).
Include negative images (glare, shadows, clean water) to cut false alarms.
Tune r, k, m, s in hotspot.py on real surveys.
Report the metrics evaluate.py prints, even if they differ from the paper's simulated 0.926 mAP / 11 FPS.
Vision only flags suspects; confirm with water-quality sensors/lab tests.
