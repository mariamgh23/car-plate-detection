import ast
import cv2
import numpy as np
import pandas as pd
import os

# ---- helper: better parsing of box strings ----
def parse_bbox(bbox_str):
    """Parse bbox strings like '[x1 y1 x2 y2]' safely."""
    if isinstance(bbox_str, str):
        # Replace multiple spaces with single comma
        clean = " ".join(bbox_str.replace("[", "").replace("]", "").split())
        nums = list(map(float, clean.split()))
        if len(nums) == 4:
            return nums
    return [0, 0, 0, 0]


# ---- helper: stylish corner borders ----
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=4, ratio=0.2):
    x1, y1 = map(int, top_left)
    x2, y2 = map(int, bottom_right)

    line_len_x = int((x2 - x1) * ratio)
    line_len_y = int((y2 - y1) * ratio)

    # TL
    cv2.line(img, (x1, y1), (x1 + line_len_x, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len_y), color, thickness)
    # TR
    cv2.line(img, (x2, y1), (x2 - line_len_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len_y), color, thickness)
    # BL
    cv2.line(img, (x1, y2), (x1 + line_len_x, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len_y), color, thickness)
    # BR
    cv2.line(img, (x2, y2), (x2 - line_len_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len_y), color, thickness)
    return img


# ---- load data ----
results = pd.read_csv("./test_interpolated.csv")

video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("./out.mp4", fourcc, fps, (width, height))

# ---- prepare license plate crops ----
license_plate = {}
for car_id in np.unique(results["car_id"]):
    sub = results[results["car_id"] == car_id]
    if sub.empty:
        continue

    max_score = sub["license_number_score"].max()
    best_row = sub[sub["license_number_score"] == max_score].iloc[0]

    frame_idx = int(best_row["frame_nmr"])
    bbox = parse_bbox(best_row["license_plate_bbox"])
    x1, y1, x2, y2 = map(int, bbox)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue

    # crop safely within frame
    y1, y2 = np.clip([y1, y2], 0, frame.shape[0])
    x1, x2 = np.clip([x1, x2], 0, frame.shape[1])

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # standardize height
    h, w = crop.shape[:2]
    new_h = 200
    new_w = int(w * (new_h / h))
    crop = cv2.resize(crop, (new_w, new_h))

    license_plate[car_id] = {
        "license_crop": crop,
        "license_plate_number": str(best_row["license_number"]),
    }


# ---- visualize per frame ----
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    df_frame = results[results["frame_nmr"] == frame_idx]
    if df_frame.empty:
        out.write(frame)
        continue

    for _, row in df_frame.iterrows():
        car_id = row["car_id"]
        car_box = parse_bbox(row["car_bbox"])
        lp_box = parse_bbox(row["license_plate_bbox"])

        cx1, cy1, cx2, cy2 = map(int, car_box)
        lx1, ly1, lx2, ly2 = map(int, lp_box)

        # draw car box
        draw_border(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), thickness=8, ratio=0.2)

        # draw license plate box
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 4)

        # draw license text + crop
        if car_id in license_plate:
            lp_crop = license_plate[car_id]["license_crop"]
            lp_text = license_plate[car_id]["license_plate_number"]

            h, w = lp_crop.shape[:2]
            text_scale = max(1.2, w / 300)
            text_thick = int(text_scale * 3)

            # overlay region (above car)
            overlay_y2 = max(cy1 - 20, h + 20)
            overlay_y1 = overlay_y2 - h
            overlay_x1 = max(cx1, 0)
            overlay_x2 = min(cx1 + w, frame.shape[1])

            if overlay_y1 > 0 and overlay_x2 - overlay_x1 > 10:
                resized_crop = cv2.resize(lp_crop, (overlay_x2 - overlay_x1, overlay_y2 - overlay_y1))
                frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = resized_crop

                cv2.rectangle(frame, (overlay_x1, overlay_y1 - 60), (overlay_x2, overlay_y1), (255, 255, 255), -1)
                cv2.putText(
                    frame,
                    lp_text,
                    (overlay_x1 + 10, overlay_y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (0, 0, 0),
                    text_thick,
                    cv2.LINE_AA,
                )

    out.write(frame)
    # uncomment to view live
    # frame_small = cv2.resize(frame, (1280, 720))
    # cv2.imshow("frame", frame_small)
    # if cv2.waitKey(1) & 0xFF == 27:  # esc
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
