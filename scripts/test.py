import torch
import numpy as np
#################################### For Image ####################################
# from PIL import Image
# from sam3.model_builder import build_sam3_image_model, _load_checkpoint
# from sam3.model.sam3_image_processor import Sam3Processor
# # Load the model
# model = build_sam3_image_model(load_from_HF=False, checkpoint_path="/data0/guojia/.cache/modelscope/models/facebook/sam3/sam3.pt")
# processor = Sam3Processor(model)
# # Load an image
# image = Image.open("./fish.jpg")
# inference_state = processor.set_image(image)
# # Prompt the model with text
# output = processor.set_text_prompt(state=inference_state, prompt="fish")

# # Get the masks, bounding boxes, and scores
# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# print(f"masks: {masks.shape}")
# print(f"boxes: {boxes.shape}")
# print(f"scores: {scores.shape}")
# print("top scores:", scores[0, :5].tolist() if scores.ndim > 1 else scores[:5].tolist())

# if masks is not None and masks.numel() > 0:
# 	mask = masks[0, 0].detach().cpu().numpy()
# 	mask_img = Image.fromarray((mask * 255).astype(np.uint8))
# 	mask_img.save("./mask0.jpg")
# 	print("saved mask to ./mask0.jpg")

#################################### For Video ####################################

import cv2
from sam3.model_builder import build_sam3_video_predictor

checkpoint_path = "/data0/guojia/.cache/modelscope/models/facebook/sam3/sam3.pt"
video_predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
video_path = "./run1.mp4"  # a JPEG folder or an MP4 video file

# Start a session
start_response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
prompt_response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=start_response["session_id"],
        frame_index=0,  # Arbitrary frame index
        text="ink",
    )
)

# Propagate prompts and collect outputs per frame
outputs_by_frame = {}
for item in video_predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=start_response["session_id"],
        propagation_direction="forward",
        start_frame_index=0,
    )
):
    outputs_by_frame[item["frame_index"]] = item["outputs"]

# Read video and write masked output
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("./run1testoutput2.mp4", fourcc, fps, (width, height))

color_map = {}
rng = np.random.default_rng(0)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out = outputs_by_frame.get(frame_idx)
    if out is not None and out.get("out_binary_masks") is not None:
        masks = out["out_binary_masks"]
        obj_ids = out.get("out_obj_ids", [])
        if masks.size > 0:
            for mask, obj_id in zip(masks, obj_ids):
                if obj_id not in color_map:
                    color_map[obj_id] = rng.integers(0, 255, size=3, dtype=np.uint8)
                color = color_map[obj_id]
                overlay = frame.copy()
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * 0.4
                    + color.astype(np.float32) * 0.6
                ).astype(np.uint8)
                frame = overlay
    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print("saved masked video to ./run1testoutput.mp4")

