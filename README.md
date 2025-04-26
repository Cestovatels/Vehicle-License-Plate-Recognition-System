## 🧠 Vehicle-License-Plate-Recognition-System

A complete vehicle and license plate recognition system with detection, tracking, OCR, and visualization.


---

## 🚀 Features

- Vehicle Detection: Detects vehicles using a YOLO-based model.
- Vehicle Tracking: Tracks detected vehicles using the SORT algorithm.
- License Plate Detection: Identifies license plate regions.
- OCR (Plate Reading): Reads license plate text from detected plates.
- Missing Frame Interpolation: Fills in missing detections using linear interpolation.
- Result Saving: Saves all detection and tracking results into a CSV file.
- Visualization: Draws tracking and recognition results on the video.

---
## 🖼 A still from the generated output video.
![A still from the generated output video.](https://github.com/Cestovatels/Vehicle-License-Plate-Recognition-System/blob/main/image/Plate-Detection-frame.png)


## 📦 Installation

```bash
Python 3.10.11 or higher is required.
git clone https://github.com/Cestovatels/Vehicle-License-Plate-Recognition-System.git
cd Vehicle-License-Plate-Recognition-System
pip install -r requirements.txt
```


## 🧰 Usage
```bash
python main.py \
  --video_path data/sample.mp4 \
  --output_dir output \
  --vehicle_model models/yolov8n.pt \
  --plate_model models/license_plate_detector.pt \
  --skip_visualization  (action="store_true")
```

## ✅ For all available arguments, check:

```bash
python main.py --help
```

## 📁 Directory Structure
```bash
Vehicle-License-Plate-Recognition-System/
    ├── model/
    │   ├── yolov8n.pt
    │   ├── license_plate_detector.pt
    ├── data/
    │   ├── sample.mp4
    ├── output/
        ├── results.csv
        ├── results_interpolated.csv
        ├── sample_output.mp4
    ├── sort/
```

## 🚀 Acknowledgements

This project was developed with the help of the following open-source repositories:

- [Automatic License Plate Recognition using YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8):  
  I used this project as a reference for implementing the license plate detection and recognition component.
  Their approach to leveraging YOLOv8 models for accurate plate detection was very helpful in building my system.

- [Simple Online and Realtime Tracking (SORT)](https://github.com/abewley/sort):  
  I integrated the SORT algorithm for object tracking to maintain consistent identification of license plates across video frames.
  This repository provided an efficient and easy-to-implement tracking solution.

Special thanks to the authors of these projects for sharing their work openly!




## 🤝 Contributions
Feel free to open an issue or submit a pull request. Feature requests and feedback are always welcome!
