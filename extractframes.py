import cv2
from pathlib import Path

video_path = Path("/data0/guojia/work_huang/sam3/run1.mp4")
out_dir = video_path.parent
stem = video_path.stem

cap = cv2.VideoCapture(str(video_path))
frames = []
for i in range(6):
	ok, frame = cap.read()
	if not ok:
		break
	out_path = out_dir / f"{stem}_{i:03d}.jpg"
	cv2.imwrite(str(out_path), frame)

cap.release()

