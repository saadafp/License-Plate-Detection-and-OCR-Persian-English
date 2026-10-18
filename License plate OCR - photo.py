#region import
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
from utils.utils import detection_info, plate_vehicle_match, crop_plate
#endregion

#region variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BEST_YOLO_PATH = BASE_DIR / "model" / "best.pt"
DATA_DIR = BASE_DIR / "data" / "img" / "english license plates" / "1.jpg"
IMG_NAME = "1"
CLASSES = [2, 3, 5, 7]    # car, motorcycle, bus, truck
SAVE_DIR = BASE_DIR / "output" / "plate_crops"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
#endregion

#region loading models
vehicle_detector_model = YOLO("yolo11n.pt")
plate_detector_model = YOLO(str(BEST_YOLO_PATH))
reader = easyocr.Reader(["en"], gpu=False)
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

#region Step 3 | Plate Detection
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

    # Step 6 | EasyOCR
    results = reader.readtext(crop_resized)

    # see if it matches to any vehicle
    match = plate_vehicle_match(vehicle_detections=info, plate_detections=[plate])
    if match:
        for m in range(len(match)):
            vehicle_id = match[m]["vehicle_id"]
            print(f"\nEasyOCR Result for Vehicle {vehicle_id}:")

            # display
            display_img = crop_resized.copy()
            for (bbox, text, conf) in results:
                print(f"Text: {text} | Confidence: {conf:.2f}")
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
                cv2.putText(display_img, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(f"English OCR - Vehicle {vehicle_id}", display_img)
                cv2.waitKey(0)

                # saving the result
                filename = f"v{vehicle_id}.png"
                cv2.imwrite(str(SAVE_DIR / filename), display_img)

    else:
        # display
        display_img = crop_resized.copy()
        for (bbox, text, conf) in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
            cv2.putText(display_img, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow(f"English OCR - Vehicle {IMG_NAME}", display_img)
            cv2.waitKey(0)

            # saving the result
            filename = f"v{IMG_NAME}.png"
            cv2.imwrite(str(SAVE_DIR / filename), display_img)

    cv2.destroyAllWindows()


#endregion
