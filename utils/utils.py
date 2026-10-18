#region imports
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
#endregion


#region pre-process
def crop_plate(frame, bbox, pad=0.08):
    """Crop a bounding box from a frame, with optional proportional padding."""
    if frame is None or frame.size == 0 or bbox is None:
        return np.empty((0, 0, 3), dtype=np.uint8)

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [int(float(value)) for value in bbox]

    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    pad_x = int(box_width * pad)
    pad_y = int(box_height * pad)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_width, x2 + pad_x)
    y2 = min(frame_height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return frame[y1:y2, x1:x2].copy()

def resize_frame(frame, max_pxl=200):
    """Upscale small crops toward max_pxl pixels high without forcing a minimum scale."""
    if frame is None or frame.size == 0:
        return frame
    if max_pxl <= 0:
        raise ValueError("max_pxl must be greater than zero.")

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0 or height >= max_pxl:
        return frame.copy()

    scale = max(1, max_pxl // height)
    new_size = (max(1, width * scale), max(1, height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

def estimate_sharpness(crop, min_sharpness=80.0, tag="", frame_count=None):
    """Return whether a crop passes the Laplacian-variance sharpness threshold."""
    if crop is None or crop.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_sharp = sharpness >= min_sharpness

    if not is_sharp and tag:
        frame_label = f" frame={frame_count}" if frame_count is not None else ""
        print(f"Rejected blurry crop: {tag}{frame_label}, sharpness={sharpness:.1f}")

    return is_sharp, sharpness
#endregion


#region yolo helpers
def detection_info(detected_objects, plot=True):
    """Extract plain-Python detection records and optionally return an annotated frame."""
    info = []
    annotated_frame = None

    for result in detected_objects:
        boxes = result.boxes
        if plot:
            annotated_frame = result.plot()
            frame_height, frame_width = annotated_frame.shape[:2]
            if frame_width > 640:
                scale = 640 / frame_width
                annotated_frame = cv2.resize(annotated_frame,(640, max(1, int(frame_height * scale))),
                                             interpolation=cv2.INTER_AREA)

        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        track_ids = boxes.id.detach().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

        for bbox, confidence, class_id, track_id in zip(xyxy, confidences, class_ids, track_ids):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            info.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(confidence),
                "cls_id": int(class_id),
                "track_id": int(track_id) if track_id is not None else None,
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            })

    return info, annotated_frame
#endregion


#region post-process
def ocr_text(ocr_results, class_names):
    """Sort YOLO character detections by x-coordinate and concatenate class names."""
    character_info = []

    for result in ocr_results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        track_ids = boxes.id.detach().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

        for bbox, confidence, class_id, track_id in zip(xyxy, confidences, class_ids, track_ids):
            cls_id = int(class_id)
            char = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)

            character_info.append({
                "bbox": tuple(float(value) for value in bbox),
                "conf": float(confidence),
                "cls_id": cls_id,
                "char": str(char),
                "track_id": int(track_id) if track_id is not None else None})

    character_info.sort(key=lambda item: item["bbox"][0])
    plate_text = "".join(item["char"] for item in character_info)
    return plate_text, character_info

def is_point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def plate_vehicle_match(vehicle_detections, plate_detections):
    """Associate each plate center with the largest vehicle box containing it."""
    matches = []

    # looking for possible vehicle candidates
    for plate in plate_detections:
        plate_id = plate.get("track_id")
        plate_center = plate.get("center")
        candidate_vehicles = [vehicle for vehicle in vehicle_detections
                              if plate_center is not None and is_point_inside_bbox(plate_center, vehicle["bbox"])]

        # calculating area, the vehicle with max area is closer to the camera
        best_vehicle = max(
            candidate_vehicles,
            key=lambda vehicle: max(0, vehicle["bbox"][2] - vehicle["bbox"][0])
            * max(0, vehicle["bbox"][3] - vehicle["bbox"][1]),
            default=None)

        matches.append({"plate_id": plate_id, "vehicle_id": best_vehicle.get("track_id") if best_vehicle else None})

    return matches

def get_storage_id(vehicle_id, plate_id, frame_count, plate_index):
    if vehicle_id is not None:
        return f"vehicle_{vehicle_id}"                             # based on vehicle_id
    if plate_id is not None:
        return f"unassigned_plate_{plate_id}"                      # based on plate_id, no vehicle_id
    return f"unassigned_frame_{frame_count}_plate_{plate_index}"   # neither vehicle_id, nor plate_id: based on frame_count
#endregion


#region best-result selection
def choose_best_ocr_text(ocr_results, plate_text, frame_count, plate_id, vehicle_id, best_texts):
    """Keep the highest confidence-length score for a vehicle/plate key."""
    if not plate_text:
        return best_texts

    confidences = []
    for result in ocr_results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            confidences.extend(boxes.conf.detach().cpu().tolist())

    if not confidences:
        return best_texts

    mean_confidence = float(np.mean(confidences))
    # mean * length of chars in plate_text
    score = mean_confidence * len(plate_text)
    storage_id = get_storage_id(vehicle_id, plate_id, frame_count, 0)

    if storage_id not in best_texts or score > best_texts[storage_id]["score"]:
        best_texts[storage_id] = {
            "storage_id": storage_id,
            "vehicle_id": vehicle_id,
            "plate_id": plate_id,
            "frame_count": frame_count,
            "text": plate_text,
            "ocr_conf": mean_confidence,
            "score": score,
        }

    return best_texts


def choose_best_plate(crop, annotated_crop, plate_text, ocr_conf, frame_count,best_plates, storage_id,
                      plate_id=None, vehicle_id=None,plate_conf=0.0, min_sharpness=80.0, sharpness=None):
    """Keep the best sharp, readable crop and its OCR text as one consistent record."""
    if crop is None or crop.size == 0 or not plate_text:
        return best_plates

    if sharpness is None:
        is_sharp, sharpness = estimate_sharpness(crop, min_sharpness=min_sharpness)
        if not is_sharp:
            return best_plates
    elif sharpness < min_sharpness:
        return best_plates

    ocr_conf = float(ocr_conf)
    score = sharpness * max(ocr_conf, 0.01)
    current = best_plates.get(storage_id)

    if current is None or score > current["score"]:
        best_plates[storage_id] = {
            "storage_id": storage_id,
            "vehicle_id": vehicle_id,
            "plate_id": plate_id,
            "crop": crop.copy(),
            "annotated_crop": annotated_crop.copy() if annotated_crop is not None else crop.copy(),
            "sharpness": sharpness,
            "ocr_conf": ocr_conf,
            "plate_conf": float(plate_conf),
            "text": plate_text,
            "frame_count": frame_count,
            "score": score,
        }

    return best_plates


def annotate_easyocr_crop(crop, ocr_results):
    """Draw EasyOCR polygons and recognized text on a copy of the crop."""
    if crop is None or crop.size == 0:
        return crop

    annotated_crop = crop.copy()
    for bbox, text, confidence in ocr_results:
        points = np.asarray(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_crop, [points], True, (0, 255, 0), 2)
        x, y = points[0, 0]
        label = f"{text} ({float(confidence):.2f})"
        cv2.putText(
            annotated_crop, label, (int(x), max(15, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    return annotated_crop


def choose_best_english_plate(crop, ocr_results, storage_id, frame_count,best_plates, plate_conf, min_sharpness=80.0,
                              plate_id=None, vehicle_id=None, annotate_crop=None,sharpness=None):
    """Normalize EasyOCR output and store the best crop and text together."""
    valid_results = [
        item for item in ocr_results  #item[0]:bbox, item[1]:text, item[3]:conf
        if len(item) >= 3 and str(item[1]).strip() and np.isfinite(float(item[2]))]

    if not valid_results:
        return best_plates

    valid_results.sort(key=lambda item: min(point[0] for point in item[0]))
    plate_text = " ".join(str(item[1]).strip() for item in valid_results)
    ocr_conf = float(np.mean([float(item[2]) for item in valid_results]))
    annotation_source = annotate_crop if annotate_crop is not None else crop
    annotated_crop = annotate_easyocr_crop(annotation_source, valid_results)

    return choose_best_plate(
        crop=crop,
        annotated_crop=annotated_crop,
        plate_text=plate_text,
        ocr_conf=ocr_conf,
        frame_count=frame_count,
        best_plates=best_plates,
        storage_id=storage_id,
        plate_id=plate_id,
        vehicle_id=vehicle_id,
        plate_conf=plate_conf,
        min_sharpness=min_sharpness,
        sharpness=sharpness,
    )


def save_easyocr_results(ocr_results, crop_resized, save_path=None):
    """Annotate EasyOCR output; save only when an explicit destination is supplied."""
    annotated_crop = annotate_easyocr_crop(crop_resized, ocr_results)
    if save_path is None:
        return annotated_crop

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), annotated_crop):
        raise IOError(f"Could not save annotated OCR crop: {save_path}")
    return save_path
#endregion


#region save image and csv
def saving_to_csv(detection_info_list, csv_name, frame_count):
    """Append records to a CSV, preserving any frame_count already in each record."""
    csv_path = Path(csv_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not detection_info_list:
        return csv_path

    rows = []
    for detection in detection_info_list:
        row = {}
        for key, value in detection.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple, dict)):
                value = str(value)
            row[key] = value
        row.setdefault("frame_count", frame_count)
        rows.append(row)

    data = pd.DataFrame(rows)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    data.to_csv(
        csv_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )
    return csv_path


def saving_cropped_image(best_plates, save_dir, plot=True):
    """Save raw and annotated versions of each selected plate crop."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for storage_id, data in best_plates.items():
        safe_id = str(storage_id).replace("/", "_").replace("\\", "_")
        raw_path = save_dir / f"{safe_id}_best.png"
        annotated_path = save_dir / f"{safe_id}_best_annotated.png"

        raw_crop = data.get("crop")
        annotated_crop = data.get("annotated_crop", raw_crop)
        if raw_crop is not None and raw_crop.size > 0:
            if not cv2.imwrite(str(raw_path), raw_crop):
                print(f"Could not save image: {raw_path}")
                continue
            saved_paths.append(raw_path)

        if annotated_crop is not None and annotated_crop.size > 0:
            if not cv2.imwrite(str(annotated_path), annotated_crop):
                print(f"Could not save annotated image: {annotated_path}")

        if plot and raw_crop is not None and raw_crop.size > 0:
            cv2.imshow(f"{storage_id} - raw", raw_crop)
            if annotated_crop is not None and annotated_crop.size > 0:
                cv2.imshow(f"{storage_id} - annotated", annotated_crop)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    if plot:
        cv2.destroyAllWindows()

    return saved_paths
#endregion
