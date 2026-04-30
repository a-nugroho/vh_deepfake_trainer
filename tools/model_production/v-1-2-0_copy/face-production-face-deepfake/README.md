# face-deepfake

This repo contains package for face deep fake prediction.

**Please read the sections below on the proper usage of this package.**

## Dependencies

The dependencies to run this package are included in `requirements.txt`.

However, the predict method in this package may need argument from 
face detection output, which comes from package `face-detection`. 
See the repo [here](https://github.com/verihubs/face-production-face-detection).

## Cloning repo
To properly clone and contribute to this repository, make sure you have Git LFS installed and initialized on your machine. Otherwise, files in MANIFEST.in may not download correctly.

### 🔧 Setup Instructions

1. **Install Git LFS**  
   Follow the official instructions: https://git-lfs.github.com/

   Or use a package manager:

   - **macOS (Homebrew)**
     ```bash
     brew install git-lfs
     ```

   - **Ubuntu/Debian**
     ```bash
     sudo apt install git-lfs
     ```

   - **Windows (Chocolatey)**
     ```powershell
     choco install git-lfs
     ```

2. **Initialize Git LFS** (only once per machine)
   ```bash
   git lfs install

Clone with
```git clone https://github.com/yourusername/your-repo.git```
OR
```git lfs pull```
if already cloned without LFS

## Installation

You can use `pip` to directly install the package with it's requirements.

```bash
python -m pip install /path/to/face-deepfake/
```

## How to Use

### From Original Image (Full/Un-cropped)
Assuming the source image is raw/full/uncropped, **you have to detect the faces 
first** using `face-detection` package. The cropped image must be a square loose-cropped face image, with 1.3 loose factor of the original face bounding box.
```python
import cv2
from face_detection import FaceDetection
from face_deepfake import DeepfakeDetection

fd = FaceDetection()
fdd = DeepfakeDetection()

# read sample image from file
img = cv2.imread("samples/1.jpeg")

# get face detection result
dets, ang = fd.predict(img)

# crop the image
img_cropped, bbox = fd.crop_single_face_square(
    img, dets, ang, loose_factor=1.3, crop_size=None
)

# get deepfake score
deepfake_score = fdd.predict(img_cropped)

# get deepfake decision
is_deepfake = fdd.classify_predictions(deepfake_score)
```

### From Cropped Image

The cropped image must be a square loose-cropped face image, 
with 1.3 loose factor of the original face bounding box.

When you have a properly cropped face image, use `predict` 
method to directly get spoof score.

```python
import cv2
from face_deepfake import DeepfakeDetection

fdd = DeepfakeDetection()

img_cropped = cv2.imread("samples/1.jpeg")

# get deepfake score
deepfake_score = fdd.predict(img_cropped)

# get deepfake decision
is_deepfake = fdd.classify_predictions(deepfake_score)
```

## Additional Infos

### Inference Speed

The inference speed depends on the size of the image passed. 


On turbo-vision2 
The inference speed of 357x467 (HxW) cropped face image excluding detection is `550 ms ± 35.8 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)`.
