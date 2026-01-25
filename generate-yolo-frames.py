import os
import numpy as np
import torch
from ultralytics import YOLO
from dataloaders.data_utils.lmdb_utils import LMDBUtility

# ---------------- CONFIG ----------------
LMDB_VIDEO_DIR = "/data/sign2gpt/Sign2GPT/data/phoenix2014t/lmdb_videos"
OUTPUT_DIR = "/data/sign2gpt/Sign2GPT/data/phoenix2014t/yolo_boxes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_MODEL = "yolov8n.pt"
DEVICE = "cuda"
# ----------------------------------------

yolo = YOLO(YOLO_MODEL).to(DEVICE)
yolo.eval()

video_names = os.listdir(LMDB_VIDEO_DIR)

for video_name in video_names:
    print(f"[YOLO] Processing {video_name}")
    lmdb_util = LMDBUtility(f"{LMDB_VIDEO_DIR}/{video_name}")
    num_frames = lmdb_util.details["num_frames"]

    boxes_per_frame = {}

    for idx in range(num_frames):
        frame = lmdb_util.get_frames([idx])[0]

        with torch.no_grad():
            results = yolo(frame, verbose=False)

        person_box = None
        for box in results[0].boxes:
            if yolo.names[int(box.cls)] == "person":
                person_box = box.xyxy[0].cpu().numpy().astype(int)
                break

        boxes_per_frame[idx] = person_box  # None if not detected

    np.save(
        f"{OUTPUT_DIR}/{video_name}.npy",
        boxes_per_frame,
        allow_pickle=True,
    )

print("✅ YOLO preprocessing complete")