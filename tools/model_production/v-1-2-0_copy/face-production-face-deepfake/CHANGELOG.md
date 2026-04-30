# CHANGELOG
## [1.2.0] - 28-01-2026
- Remove ConvNext base model
- Add two new base models
- Add logits tuning

## [1.1.0] - 22-08-2025
- Add new base model
- Add three-model based Random Forest logit ensembler model

### Changed
- Updated `DeepfakeDetection` class to use Random Forest logic ensembler model
  - Updated `predict` function to ensemble predictions from `DeepfakeEngine`, `CLIPEngine`, and `DeepfakeEngineEffort`

### HOTFIX
- 29-08-2025: Use logit averaging in `DeepfakeDetection`
  
## [1.0.0] - 07-03-2025
### Added
- Changed flow to use ensemble model for improved deepfake detection
- Refactored code for better organization and maintainability
- Added new code to the following files:
  - `face_deepfake/main.py`
    - Updated to use ensemble model
  - `face_deepfake/engine_clip.py` and `face_deepfake/engine_clip_onnx.py`
    - Added `CLIPEngine` to use PyTorch or ONNX model CLIP instance
  - `face_deepfake/network/clip.py`
    - Added clip model architecture
- Removed unnecessary code from `face_deepfake/utils/crop.py` (file deleted)

### Changed
- Updated `DeepfakeDetection` class to use ensemble model
  - Updated `predict` function to ensemble predictions from `DeepfakeEngine` and `CLIPEngine`
  - Deleted functions `check_valid_dets`, `predict_cropped`, and `classify_main_score`
  - Updated function `classify_prediction` to directly return the result

- Updated `DeepfakeEngine` class in `face_deepfake/engine.py` to use PyTorch or ONNX model instance
  - Updated `_preprocess_data` method to perform several preprocessing steps on the input image
  - Updated `_run_inference` method to run the inference and return logits score
  - Updated `predict` method to predict deepfake score from the given image
  - Deleted checking detections, rotate, and crop face functionality

## [0.1.0] - 13-07-2023

### Added

- Initial release of face deepfake model
