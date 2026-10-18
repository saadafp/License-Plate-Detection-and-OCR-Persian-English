#region imports
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from utils.utils import (detection_info,plate_vehicle_match,crop_plate,resize_frame,estimate_sharpness,
                         ocr_text,get_storage_id,choose_best_plate,saving_to_csv,saving_cropped_image)
#endregion


#region variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BEST_YOLO_PATH = BASE_DIR / "model" / "best.pt"
BEST_OCR_PATH = BASE_DIR / "model" / "OCR_weights" / "best (1).pt"
SAVE_DIR = BASE_DIR / "output" / "video_outputs" / "plate_crops_persian"
DATA_DIR = BASE_DIR / "data" / "2 cars.mp4"

VEHICLE_CSV_PATH = SAVE_DIR / "vehicle_detections.csv"
PLATE_CSV_PATH = SAVE_DIR / "plate_detections.csv"
OCR_CSV_PATH = SAVE_DIR / "best_ocr_texts.csv"

FRAME_SKIP = 1
MIN_PLATE_HEIGHT = 20
CLASSES = [2, 3, 5, 7]
MIN_SHARPNESS = 80.0
VEHICLE_CONF = 0.5
PLATE_CONF = 0.5
OCR_CONF = 0.3
# endregion


#region loading model
VEHICLE_DETECTOR_MODEL = YOLO("yolo11n.pt")
PLATE_DETECTOR_MODEL = YOLO(BEST_YOLO_PATH)
PERSIAN_OCR_MODEL = YOLO(BEST_OCR_PATH)
#endregion


#region functions
def license_plate_reader(source=DATA_DIR,vehicle_detection_model=VEHICLE_DETECTOR_MODEL,plate_detection_model=PLATE_DETECTOR_MODEL,
                        persian_ocr_model=PERSIAN_OCR_MODEL,classes=CLASSES,frame_skip=FRAME_SKIP,min_plate_height=MIN_PLATE_HEIGHT,
                         min_sharpness=MIN_SHARPNESS,vehicle_csv_path=VEHICLE_CSV_PATH,plate_csv_path=PLATE_CSV_PATH,
                         ocr_csv_path=OCR_CSV_PATH,save_dir=SAVE_DIR,display=True):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {source}")

    frame_count = 0
    best_plates = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                       continue

            # 1. Vehicle detection and tracking.
            detected_vehicles = vehicle_detection_model.track(source=frame, imgsz=640, conf=VEHICLE_CONF,
                                                            classes=classes, persist=True, verbose=False)
            vehicle_detections, vehicle_plot = detection_info(detected_vehicles, plot=display)
            saving_to_csv(vehicle_detections, vehicle_csv_path, frame_count)

            # 2. Plate detection and tracking.
            detected_plates = plate_detection_model.track(source=frame, imgsz=640, conf=PLATE_CONF,persist=True, verbose=False)
            plate_detections, plate_plot = detection_info(detected_plates, plot=display)
            saving_to_csv(plate_detections, plate_csv_path, frame_count)

            # 3. Associate plate tracks with vehicle tracks.
            matches = plate_vehicle_match(vehicle_detections, plate_detections)
            plate_to_vehicle = {item["plate_id"]: item["vehicle_id"]
                                for item in matches if item["plate_id"] is not None}

            # 4. Crop, filter and recognize each plate.
            for plate_index, plate in enumerate(plate_detections):
                crop = crop_plate(frame, plate["bbox"])
                if crop.size == 0 or crop.shape[0] < min_plate_height:
                    continue

                plate_id = plate["track_id"]
                vehicle_id = plate_to_vehicle.get(plate_id) if plate_id is not None else None
                storage_id = get_storage_id(vehicle_id, plate_id, frame_count, plate_index)

                is_sharp, sharpness = estimate_sharpness(crop, min_sharpness=min_sharpness,tag=storage_id, frame_count=frame_count)
                if not is_sharp:
                    continue

                crop_resized = resize_frame(crop, max_pxl=200)
                ocr_results = persian_ocr_model.predict(source=crop_resized, imgsz=640, conf=OCR_CONF, verbose=False)
                plate_text, character_info = ocr_text(ocr_results, class_names=persian_ocr_model.names)
                if not plate_text:
                    continue

                confidences = [item["conf"] for item in character_info]
                if not confidences:
                    continue
                mean_ocr_conf = float(np.mean(confidences))
                annotated_crop = ocr_results[0].plot() if ocr_results else crop_resized.copy()

                # finding the sharpest and readable crop, updating till finding the one
                choose_best_plate(
                    crop=crop,
                    annotated_crop=annotated_crop,
                    plate_text=plate_text,
                    ocr_conf=mean_ocr_conf,
                    frame_count=frame_count,
                    best_plates=best_plates,
                    storage_id=storage_id,
                    plate_id=plate_id,
                    vehicle_id=vehicle_id,
                    plate_conf=plate["conf"],
                    min_sharpness=min_sharpness,
                    sharpness=sharpness)

                print(f"Frame={frame_count}, ID={storage_id}, text={plate_text}")

            # 5. Display tracking results when requested.
            if display:
                if vehicle_plot is not None:
                    cv2.imshow("vehicle_tracking", vehicle_plot)
                if plate_plot is not None:
                    cv2.imshow("plate_tracking", plate_plot)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    # Keep text and image selected from the same best OCR sample.
    best_texts = {
        storage_id: {
            key: data[key]
            for key in ("storage_id", "vehicle_id", "plate_id", "frame_count", "text", "ocr_conf", "score")}
        for storage_id, data in best_plates.items()
        if data.get("text")}

    ocr_saved_path = saving_to_csv(list(best_texts.values()), ocr_csv_path, frame_count)
    print(f"Best OCR texts saved: {ocr_saved_path}")

    saved_plate_paths = saving_cropped_image(best_plates, save_dir=save_dir, plot=True)
    return best_plates, saved_plate_paths, best_texts
#endregion


#region main
if __name__ == "__main__":
    license_plate_reader()
#endregion
