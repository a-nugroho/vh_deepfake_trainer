from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile

from face_deepfake.network.clip import CLIPDetector

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


def load_network(network, save_filename, device):
    stdict = torch.load(save_filename, map_location=device)
    network.load_state_dict(stdict, strict=False)
    print(f"Load model in {save_filename}")
    return network


class CLIPEngine:
    INPUT_SIZE = (224, 224)
    _INPUT_MEAN = [0.5, 0.5, 0.5]
    _INPUT_STD = [0.5, 0.5, 0.5]

    def __init__(self, device_name: str = "cpu"):

        model_path = Path(__file__).parent / "weights/clip_ciplab_best.pth"

        self._model_path = model_path
        self.device_name = device_name

        self.net = CLIPDetector()

        self.net = load_network(self.net, self._model_path, self.device_name)
        self.net.train(False)
        self.net.eval()

    def _preprocess_data(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for deepfake detection.

        This function performs several preprocessing steps on the input image:
        1. Converts the image from BGR to RGB format.
        2. Resizes the image to the specified input size.
        3. Converts the image to a tensor with the channel-first format.
        4. Normalizes the image tensor using the provided mean and standard deviation.
        5. Expands the dimensions of the image tensor to add a batch dimension.
        6. Converts the image tensor to a PyTorch tensor.

        Args:
            img (np.ndarray): The input image in BGR format as a NumPy array.

        Returns:
            np.ndarray: The preprocessed image as a PyTorch tensor in BCHW format.
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = img.resize(self.INPUT_SIZE, Image.BILINEAR)
        img = _to_tensor(img)
        img = _normalize(
            img,
            mean=self._INPUT_MEAN,
            std=self._INPUT_STD,
        )
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)

        return img

    def _run_inference(self, inp_tensor: np.ndarray) -> float:
        """
        Run the inference.

        Args:
            inp_tensor (np.ndarray): The input tensor.

        Returns:
            float: Logits score.
        """
        prediction = self.net(inp_tensor).cpu().detach().numpy()

        return prediction

    def predict(
        self,
        img: np.ndarray,
    ) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score by preprocessing the image and
        running inference on the processed data.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Logits score indicating the likelihood of the image being a deepfake.
        """

        inp_tensor: np.ndarray = self._preprocess_data(img)
        predictions = self._run_inference(inp_tensor)

        return predictions
