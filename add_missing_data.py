import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(data):
    # convert everything to numeric once
    frame_numbers = np.array([int(row["frame_nmr"]) for row in data])
    car_ids = np.array([int(float(row["car_id"])) for row in data])
    car_bboxes = np.array([list(map(float, row["car_bbox"].strip("[]").split())) for row in data])
    license_bboxes = np.array([list(map(float, row["license_plate_bbox"].strip("[]").split())) for row in data])

    interpolated_data = []

    for car_id in np.unique(car_ids):
        # filter per car
        mask = car_ids == car_id
        frames = frame_numbers[mask]
        cars = car_bboxes[mask]
        plates = license_bboxes[mask]

        if len(frames) < 2:
            # only one frame — no interpolation possible
            continue

        # create continuous frame range
        full_frames = np.arange(frames.min(), frames.max() + 1)

        # interpolate each coordinate independently
        interp_car = interp1d(frames, cars, axis=0, kind="linear", fill_value="extrapolate")
        interp_lp = interp1d(frames, plates, axis=0, kind="linear", fill_value="extrapolate")

        car_interp_values = interp_car(full_frames)
        lp_interp_values = interp_lp(full_frames)

        for i, f in enumerate(full_frames):
            row = {"frame_nmr": int(f), "car_id": int(car_id)}
            row["car_bbox"] = " ".join(map(lambda x: f"{x:.2f}", car_interp_values[i]))
            row["license_plate_bbox"] = " ".join(map(lambda x: f"{x:.2f}", lp_interp_values[i]))

            # check if this frame existed originally
            same_frame = [p for p in data if int(p["frame_nmr"]) == f and int(float(p["car_id"])) == car_id]
            if same_frame:
                src = same_frame[0]
                row["license_plate_bbox_score"] = src.get("license_plate_bbox_score", "0")
                row["license_number"] = src.get("license_number", "0")
                row["license_number_score"] = src.get("license_number_score", "0")
            else:
                # interpolated
                row["license_plate_bbox_score"] = "0"
                row["license_number"] = "0"
                row["license_number_score"] = "0"

            interpolated_data.append(row)

    return interpolated_data


# ---- main ----
with open("test.csv", "r") as file:
    reader = csv.DictReader(file)
    data = list(reader)

interpolated = interpolate_bounding_boxes(data)

header = [
    "frame_nmr",
    "car_id",
    "car_bbox",
    "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number",
    "license_number_score",
]

with open("test_interpolated.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated)

print("✅ Interpolation complete:", len(interpolated), "rows written.")
