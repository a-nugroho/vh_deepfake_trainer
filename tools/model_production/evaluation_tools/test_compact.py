#folder_path = "samples/"
folder_path = "/mnt/hdd/workspace/adi/verihubs_repos_push/face-production-face-deepfake/evaluation_tools/excl_sample"
folder_path = "/mnt/ssd/datasets/unsorted/taspen/taspen_photos"
folder_path = "/mnt/ssd/datasets/deepfake/vh_55plus/fake/e4s"
#folder_path = "/mnt/hdd/workspace/adi/verihubs_repos_push/face-production-face-deepfake/evaluation_tools/on-call/2025-08-25/23 Deepfake"
import cv2
#from face_detection import FaceDetection
import os
os.chdir(os.path.dirname(os.getcwd()))
#os.chdir('/mnt/hdd/workspace/adi/verihubs_repos/face-production-face-deepfake')
import sys 
sys.path.append(os.getcwd())
#sys.path.append('/mnt/hdd/workspace/adi/verihubs_repos/face-production-face-deepfake')
#from face_deepfake import DeepfakeDetection
import numpy as np
#from dataset.deepfake_dataset import DeepFakeDataset
import torch
import yaml
from tqdm import tqdm
#from face_deepfake.engine_onnx import DeepfakeEngine, DeepfakeEngineEffort
#from face_deepfake.engine_clip_onnx import CLIPEngine
#from sklearn import metrics
from face_detection import FaceDetection
from face_detection.utils import align_utils
from skimage import transform as trans
from face_deepfake import DeepfakeDetection
import dlib
def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def extract_aligned_face_dlib(face_detector, predictor, image, res=224, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        landmark = predictor(cropped_face, face_align[0])
        landmark = shape_to_np(landmark)

        return cropped_face, landmark,face
    
    else:
        return None, None

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def resize_and_center_content_opencv(img, target_size=224, threshold=5):
    """
    Resize an image while maintaining aspect ratio, detect content, 
    center it on a black square canvas of size (target_size x target_size).
    
    Parameters:
        img (np.ndarray): Input BGR image (OpenCV format).
        target_size (int): Desired output size (square).
        threshold (int): Intensity threshold for content detection.
    
    Returns:
        np.ndarray: Processed image (target_size x target_size x 3).
    """
    # Step 1: Convert to grayscale for content detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Find non-black area using threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)

    if coords is None:
        # Image is all black, return centered black image
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]

    # Step 3: Resize while maintaining aspect ratio
    h_c, w_c = cropped.shape[:2]
    scale = target_size / max(h_c, w_c)
    resized = cv2.resize(cropped, (int(w_c * scale), int(h_c * scale)), interpolation=cv2.INTER_CUBIC)

    # Step 4: Create black canvas and center the resized image
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    h_r, w_r = resized.shape[:2]
    top = (target_size - h_r) // 2
    left = (target_size - w_r) // 2

    canvas[top:top+h_r, left:left+w_r] = resized
    return canvas


fd = FaceDetection()
fdd = DeepfakeDetection()

num_pred_deepfake = 0
num_pred_real = 0
num_real = 0
num_deepfake = 0
for img_file in sorted(os.listdir(folder_path)):
    img_path = os.path.join(os.getcwd(),folder_path,img_file)
    #try:

    # read sample image from file
    img = cv2.imread(img_path)

    # get face detection result
    dets, ang = fd.predict(img)
    #print(dets[0][:4])
    #print(dets)
    # crop the image
    img_cropped, bbox = fd.crop_single_face(
        img, dets, ang, loose_factor=1.3, crop_size=None, square=True
    )

    # get deepfake score
    deepfake_score = fdd.predict(img_cropped)
    #logits_1 = fdd.engine_1.predict(img_cropped)[0]
    #logits_2 = fdd.engine_2.predict(img_cropped)[0]
    #logits_3 = fdd.engine_3.predict(img_cropped)[0]
    
    # get deepfake decision
    is_deepfake = fdd.classify_predictions(deepfake_score)
    print(f"Sample {img_file} {'DEEPFAKE' if is_deepfake else 'REAL'}, SCORE: {deepfake_score:.4f}")
    #print(f"Logits 1: {logits_1}, Logits 2: {logits_2}, Logits 3: {logits_3}")
    #except:
    #    print(f"Cant process path {img_path}")
    num_pred_deepfake += int(is_deepfake)
    num_pred_real += int(not is_deepfake)
print(f"Pred Deepfake %: {num_pred_deepfake/(num_pred_deepfake+num_pred_real)}")
print(f"Pred Real %: {num_pred_real/(num_pred_deepfake+num_pred_real)}")