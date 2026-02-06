import os
import numpy as np
import cv2

# .npz format video path
original_path = "/data0/guojia/robotics/code/RoboticsDiffusionTransformer/datasets/clean_cook/run1/testing_demo_camera_332522077725.npz"
output_path = "/data0/guojia/work_huang/sam3/run1.mp4"


def _find_video_array(npz_data):
	priority_keys = [
		"video",
		"videos",
		"frames",
		"images",
		"rgb",
		"observations",
		"obs",
		"image",
		"imgs",
	]
	keys = list(npz_data.keys())
	for key in priority_keys:
		if key in npz_data:
			arr = npz_data[key]
			if isinstance(arr, np.ndarray) and arr.ndim >= 3:
				return key, arr
	for key in keys:
		arr = npz_data[key]
		if isinstance(arr, np.ndarray) and arr.ndim >= 3:
			return key, arr
	raise ValueError(f"No video-like array found. Available keys: {keys}")


def _to_uint8(frames):
	if frames.dtype == np.uint8:
		return frames
	if np.issubdtype(frames.dtype, np.floating):
		frames = np.clip(frames, 0.0, 1.0) if frames.max() <= 1.0 else np.clip(frames, 0.0, 255.0)
		return (frames * 255.0).astype(np.uint8) if frames.max() <= 1.0 else frames.astype(np.uint8)
	return np.clip(frames, 0, 255).astype(np.uint8)


def _normalize_frames(frames):
	if frames.ndim == 3:
		# (T, H, W) -> (T, H, W, 1)
		frames = frames[:, :, :, None]
	if frames.ndim != 4:
		raise ValueError(f"Unsupported frame shape: {frames.shape}")

	# Handle channel-first (T, C, H, W)
	if frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
		frames = np.transpose(frames, (0, 2, 3, 1))

	# Drop alpha channel if present
	if frames.shape[-1] == 4:
		frames = frames[..., :3]
	return frames


def _get_fps(npz_data, default=5.0):
	for key in ["fps", "frame_rate", "video_fps"]:
		if key in npz_data:
			try:
				return float(npz_data[key])
			except Exception:
				pass
	return default


def main():
	npz_data = np.load(original_path, allow_pickle=True)
	key, frames = _find_video_array(npz_data)
	fps = _get_fps(npz_data)

	frames = _normalize_frames(frames)
	frames = _to_uint8(frames)

	height, width = frames.shape[1], frames.shape[2]
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

	for frame in frames:
		if frame.shape[-1] == 1:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		else:
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		writer.write(frame)

	writer.release()
	print(f"Loaded '{key}' with shape {frames.shape} and saved to {output_path}")


if __name__ == "__main__":
	main()