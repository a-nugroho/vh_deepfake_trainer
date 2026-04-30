from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _to_tensor(img: Image.Image) -> np.ndarray:
    """
    Converts a PIL image to a numpy array, then transposes the array to put the channels
    first, and finally divides the array by 255.0 to normalize the values.

    :param img: The image to be transformed
    :return: A numpy array like tensor of the image
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    img = np.asarray(img, dtype=np.float32)
    img = np.ascontiguousarray(img.transpose(2, 0, 1)) / 255.0

    return img


def _normalize(ten: np.ndarray, mean, std):
    """
    Normalize the given image tensor by subtracting with mean, and division with
    std.
    The image should be in CHW shape and normalized to 0-1 range.

    Args:
        ten (np.ndarray): The image tensor
        mean (tuple): 3-element sequence containing mean for R, G,
            and B channels respectively.
        std (tuple): 3-eleement sequence containing std for R, G, and B channels
            respectively.

    Returns:
        np.ndarray: The normalized image tensor.
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    return (ten - mean) / std


class DeepfakeEngine:
    DEFAULT_THRESHOLD = 0.50
    
    def __init__(self, device_name: str = "cpu", 
                 input_size:tuple =  (224,224), 
                 input_mean:tuple = (0.485, 0.456, 0.406),
                 input_std:tuple = (0.229, 0.224, 0.225),
                 model_path:str = "",
                 logit_offset:tuple = (0, 0),
                 logit_temp:float = 1.0
                 ):
        """
        Initialize the DeepfakeEngine with the specified device.

        Args:
            device_name (str, optional): The device to use for inference. Defaults to "cpu".
                Options are "cpu" for CPUExecutionProvider and "cuda" for CUDAExecutionProvider.

        This initializes an ONNX inference session with the model specified by the model path.
        If CUDA is not available or fails, it defaults to using the CPU.
        """

        model_path = f"{Path(__file__).parent}/weights/{model_path}"
        
        self._model_path = model_path
        self.input_size = input_size
        self.input_mean = input_mean
        self.input_std = input_std
        self.logit_offset = np.array(logit_offset)
        self.logit_temp = np.array(logit_temp)
        if device_name == "cpu":
            self.net = ort.InferenceSession(
                self._model_path, providers=["CPUExecutionProvider"]
            )
        else:
            try:
                self.net = ort.InferenceSession(
                    self._model_path, providers=["CUDAExecutionProvider"]
                )
            except:
                self.net = ort.InferenceSession(
                    self._model_path, providers=["CPUExecutionProvider"]
                )

    def _preprocess_data(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """
        Preprocess the input image for deepfake detection.

        This function performs several preprocessing steps on the input image:
        1. Converts the image from BGR to RGB format.
        2. Resizes the image to the specified input size.
        3. Converts the image to a tensor with the channel-first format.
        4. Normalizes the image tensor using the provided mean and standard deviation.
        5. Expands the dimensions of the image tensor to add a batch dimension.

        Args:
            img (np.ndarray): The input image in BGR format as a NumPy array.

        Returns:
            np.ndarray: The preprocessed image as a NumPy array in BCHW format.
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = img.resize(self.input_size, resample=Image.BILINEAR)
        img = _to_tensor(img)
        img = _normalize(
            img,
            mean=self.input_mean,
            std=self.input_std,
        )
        img = np.expand_dims(img, axis=0)

        return img

    def _run_inference(self, inp_tensor: np.ndarray) -> float:
        """Run the inference.

        Args:
            inp_tensor (np.ndarray): The input tensor.

        Returns:
            float: Logits score.
        """

        predictions = self.net.run(output_names=None, input_feed={"input": inp_tensor})

        return predictions

    def predict(
        self,
        img: np.ndarray,
    ) -> float:
        """
        Predict deepfake score from given image.

        Args:
            img (np.ndarray): The input image in BGR format as a NumPy array.

        Returns:
            float: Logits score.
        """
        inp_tensor: np.ndarray = self._preprocess_data(img)
        predictions = self._run_inference(inp_tensor)
        predictions = self.scaling_logits(predictions,self.logit_offset,self.logit_temp)
        return predictions

    def scaling_logits(self,logits,offset,temp):
        logits = logits+offset
        logits = logits/temp
        return logits
  
class SKLearnEnsembler:
    def __init__(self, device_name: str = "cpu", weight_name: str = "deepfake_detection_model.onnx"):
        model_path = (
            Path(__file__).parent / f"weights/{weight_name}"
        )

        self._model_path = model_path
        if device_name == "cpu":
            self.model = ort.InferenceSession(
                self._model_path, providers=["CPUExecutionProvider"]
            )
        else:
            try:
                self.model = ort.InferenceSession(
                    self._model_path, providers=["CUDAExecutionProvider"]
                )
            except:
                self.model = ort.InferenceSession(
                    self._model_path, providers=["CPUExecutionProvider"]
                )
        
    def _run_inference(self, inp_tensor: np.ndarray) -> float:
        input_name = self.model.get_inputs()[0].name
        predictions = self.model.run(output_names=None, input_feed={input_name: inp_tensor})

        return predictions

    def predict(
        self,
        inp_tensor: np.ndarray,
        ) -> float:
        predictions = self._run_inference(inp_tensor)

        return predictions