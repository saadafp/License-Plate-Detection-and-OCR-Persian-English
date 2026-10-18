#region import
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils.utils import detection_info, plate_vehicle_match, crop_plate
#endregion


#region variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BEST_YOLO_PATH = BASE_DIR / "model" / "best.pt"
BEST_OCR_PATH = BASE_DIR / "model" / "OCR_weights" / "best (1).pt"
DATA_DIR = BASE_DIR / "data" / "img" / "persian license plates" / "107.jpg"
IMG_NAME = "107"
CLASSES = [2, 3, 5, 7]
SAVE_DIR = BASE_DIR / "output" / "plate_crops_persian"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
#endregion


#region loading models
vehicle_detector_model = YOLO("yolo11n.pt")
plate_detector_model = YOLO(str(BEST_YOLO_PATH))
persian_ocr_model = YOLO(str(BEST_OCR_PATH))
#endregion

#region Step 1 | Reading image
img = cv2.imread(str(DATA_DIR))
print("Image shape:", img.shape)
#endregion

#region Step 2 | Vehicle Detection
vehicle_results = vehicle_detector_model.predict(source=img, imgsz=640, conf=0.5, classes=CLASSES)[0]
info, _ = detection_info(vehicle_results, plot=True)

print("\n--- Detected Vehicles ---")
for vehicle in info:
    print(f"conf={vehicle['conf']:.2f} | class={vehicle['cls_id']} | center={vehicle['center']}")
#endregion


#region Step 3 |Plate Detection
plate_results = plate_detector_model.predict(source=img, imgsz=640, conf=0.5)[0]
info_plate, _ = detection_info(plate_results, plot=True)

print("\n--- Detected Plates ---")
for plate in info_plate:
    print(f"conf={plate['conf']:.2f} | center={plate['center']}")
#endregion


#region Step 4 | Matching
matches = plate_vehicle_match(vehicle_detections=info, plate_detections=info_plate)
print("\n--- Plate → Vehicle Association ---")
if not matches:
    print("No association found.")
else:
    for m in matches:
        print(f"Plate ID={m['plate_id']} → Vehicle ID={m['vehicle_id']}")
#endregion

#region Step 5 | Crop + OCR
for plate in info_plate:
    crop = crop_plate(img, plate["bbox"])
    if crop.size == 0 or crop.shape[0] < 20:
        continue

    # magnifying the cropped image. max 200pxl
    h, w = crop.shape[:2]
    scale = max(3, 200 // max(h, 1))
    crop_resized = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Step 6 |YOLO Persian OCR
    ocr_results = persian_ocr_model.predict(source=crop_resized, imgsz=640, conf=0.25)

    print(f"\nPersian YOLO OCR Result:")
    for result in ocr_results:
        if result.boxes is not None:
            for box in result.boxes:
                print(f"Class ID: {int(box.cls[0])} | Confidence: {float(box.conf[0]):.2f}")

        annotated = result.plot()
        if annotated.shape[0] < 150:
            annotated = cv2.resize(annotated, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        # let's see if it matches any vehicles
        match = plate_vehicle_match(vehicle_detections=info, plate_detections=[plate])
        if match:
            # accessing the value of key:"vehicle_id"
            for m in range(len(match)):
                vehicle_id = match[m]["vehicle_id"]

                # saving the result
                filename = f"v{vehicle_id}.png"
                cv2.imwrite(str(SAVE_DIR / filename), annotated)

                # display
                cv2.imshow(f"Persian OCR - Vehicle {vehicle_id}", annotated)
                cv2.waitKey(0)

        else:
            # saving the result
            filename = f"v{IMG_NAME}.png"
            cv2.imwrite(str(SAVE_DIR / filename), annotated)

            # display
            cv2.imshow(f"Persian OCR - Vehicle {IMG_NAME}", annotated)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
#endregion




