"""Download the third-party models and test images (not stored in the repo).

  python fetch_assets.py
"""
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = {
    # MediaPipe Face Landmarker (Apache-2.0)
    "models/face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    # OpenCV Zoo YuNet face detector and SFace recognizer
    "models/face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    # Test images for the offline self-tests
    "testdata/portrait.jpg": "https://storage.googleapis.com/mediapipe-assets/portrait.jpg",
    "testdata/lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
}

for rel, url in ASSETS.items():
    path = HERE / rel
    if path.exists() and path.stat().st_size > 0:
        print(f"ok        {rel}")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download  {rel}")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
