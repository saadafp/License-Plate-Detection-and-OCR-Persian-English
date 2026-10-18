# License Plate Detection and OCR | Persian, English

A real-time license plate recognition pipeline that detects vehicles and plates, tracks them across video frames, reads plate text, and keeps the single best reading per vehicle. It supports **Persian plates** (custom YOLO character-detection model) and **English plates** (EasyOCR).

## Features

- **Vehicle detection and tracking** with YOLO11n (cars, motorcycles, buses, trucks)
- **Plate detection and tracking** with a custom-trained YOLO model
- **Plate-to-vehicle association**: each plate is linked to the largest vehicle box that contains the plate center
- **Two OCR engines**
  - Persian: custom YOLO model that detects characters, sorted left-to-right by x-coordinate
  - English: EasyOCR restricted to `A-Z` and `0-9`
- **Quality filtering**: crops that are too small or too blurry (Laplacian variance) are rejected before OCR
- **Best-result selection per vehicle**: the final plate is the crop with the highest `sharpness × OCR confidence` score, so text and image always come from the same frame
- **Structured outputs**: CSV logs and raw and annotated plate crops

## Pipeline

```
Frame
  ├─► Vehicle detection + tracking (YOLO11n)
  ├─► Plate detection + tracking (custom YOLO)
  │
  └─► Plate ↔ Vehicle matching
        └─► Crop plate → size filter → sharpness filter → upscale
              └─► OCR (Persian YOLO  |  EasyOCR)
                    └─► Keep best crop + text per vehicle
                          └─► CSV + PNG export
```

## Project Structure

```
.
├── notebooks/
│   ├── yolo11-persian-license-plate-for-ocr.ipynb
│   └── yolo11-persian-license-plate-for-detection.ipynb   
├── utils/
│   ├── __init__.py
│   └── utils.py                              # shared helpers
├── License plate OCR - photo.py              # English, single image
├── Persian license plate OCR - photo.py      # Persian, single image
├── English license plate OCR - video v1.py   # English, video
├── Persian License plate OCR - video v1.py   # Persian, video

```

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install ultralytics easyocr opencv-python numpy pandas
```

Place your trained weights in `model/` as shown above. `yolo11n.pt` is downloaded automatically by Ultralytics on first run.

## Usage

**Video**

```bash
python "English license plate OCR - video v1.py"
python "Persian License plate OCR - video v1.py"
```

Set the input path through `DATA_DIR` at the top of each script. Press `q` to stop the preview window.

**Single image**

```bash
python "License plate OCR - photo.py"
python "Persian license plate OCR - photo.py"
```

Set `DATA_DIR` and `IMG_NAME` at the top of each script.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `FRAME_SKIP` | `1` | Process every N-th frame |
| `MIN_PLATE_HEIGHT` | `20` | Minimum plate crop height (px) |
| `MIN_SHARPNESS` | `80.0` | Minimum Laplacian variance to accept a crop |
| `VEHICLE_CONF` | `0.5` | Vehicle detection confidence threshold |
| `PLATE_CONF` | `0.5` | Plate detection confidence threshold |
| `OCR_CONF` | `0.3` | Persian character confidence threshold |
| `CLASSES` | `[2, 3, 5, 7]` | COCO classes: car, motorcycle, bus, truck |

## Output

Each video run writes to `output/video_outputs/`:

- `vehicle_detections*.csv`: per-frame vehicle detections
- `plate_detections*.csv`: per-frame plate detections
- `best_ocr_texts*.csv`: best text per vehicle (frame, confidence, score)
- `<vehicle_id>_best.png` and `<vehicle_id>_best_annotated.png`: best raw and annotated plate crops

## How Best-Result Selection Works

For every tracked vehicle, each valid reading is scored as:

```
score = sharpness × OCR confidence
```

Only the highest-scoring crop and its text are kept. Plates with no vehicle ID fall back to the plate track ID, and then to the frame number.

## Results

<!-- Add 2-3 screenshots or a short GIF here -->
<!-- ![Persian demo](assets/persian_demo.png) -->
<!-- ![English demo](assets/english_demo.png) -->

## Limitations

- Plate-to-vehicle matching uses the plate center only, so heavily overlapping vehicles can be mismatched.
- Best-frame selection depends on tracker ID stability; ID switches split one vehicle into several records.
- Persian OCR reads characters by x-position only and has no plate-format validation.

## Roadmap

- [ ] Plate-format validation (regex / character rules)
- [ ] Multi-frame text voting
- [ ] GPU / TensorRT optimization for real-time use

## Tech Stack

Python · Ultralytics YOLO · EasyOCR · OpenCV · NumPy · Pandas

## License

Add your license here (e.g. MIT).
