# Python Face Detection

A simple real-time face detection script using OpenCV.

## Requirements

- Python 3.8+ (tested with Python 3.13)
- `opencv-python`

## Install

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python
```

## Run

```bash
python face_detector.py
```

Then focus the webcam window and press `q` to exit.

## Notes

- If `cv2.CascadeClassifier` is not found, install a compatible OpenCV version:

```bash
python -m pip install --upgrade opencv-python==4.13.0.92
```

- Make sure your webcam is accessible and not used by another application.
