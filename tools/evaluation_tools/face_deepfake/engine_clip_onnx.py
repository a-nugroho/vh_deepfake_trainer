from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFile
import tritonclient.grpc as grpcclient
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


class CLIPEngine:
    INPUT_SIZE = (224, 224)
    _INPUT_MEAN = [0.5, 0.5, 0.5]
    _INPUT_STD = [0.5, 0.5, 0.5]

    def __init__(self, device_name: str = "cpu", use_triton=False):
        """
        Initialize the CLIPEngine with the specified device.

        Args:
            device_name (str, optional): The device to use for inference. Defaults to "cpu".
                Options are "cpu" for CPUExecutionProvider and "cuda" for CUDAExecutionProvider.

        This initializes an ONNX inference session with the model specified by the model path.
        If CUDA is not available or fails, it defaults to using the CPU.
        """
        model_path = Path(__file__).parent / "weights/clip_detector_optimized.onnx"
        self.use_triton = use_triton
        if use_triton:
            print(f"{self.__class__.__name__} use Triton")
            self.net = grpcclient.InferenceServerClient(
                url="grpc.aws-production.setoranku.com:443",
                ssl=True,
                root_certificates=None,  # Use system default certificates
                private_key=None,
                certificate_chain=None
            )
        else:    
            if device_name == "cpu":
                self.net = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
            else:
                try:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CUDAExecutionProvider"]
                    )
                except:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )
                    
        print(f"{self.__class__.__name__} net class: {self.net.__class__}")

    def _preproce_data(
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
        img = _normalize(img, self._INPUT_MEAN, self._INPUT_STD)
        img = np.expand_dims(img, axis=0)
        return img

    def _run_inference(self, inp_tensor: np.ndarray) -> float:
        """
        Run the inference.

        Args:
            inp_tensor (np.ndarray): The input tensor.

        Returns:
            float: Logits score.
        """
        if self.use_triton:
            inputs = []
            outputs = []
            inputs.append(grpcclient.InferInput("input", [1, 3, 224, 224], "FP32"))
            outputs.append(grpcclient.InferRequestedOutput("output"))
            inputs[0].set_data_from_numpy(inp_tensor)

        
            results = self.net.infer(
            model_name="clip-detector",
            inputs=inputs,
            outputs=outputs,
            )
            # Get the output arrays from the results
            predictions = results.as_numpy("output")
        else:
            predictions = self.net.run(output_names=None, input_feed={"input": inp_tensor})

        return predictions

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Logits score indicating the likelihood of the image being a deepfake.
        """

        inp_tensor: np.ndarray = self._preproce_data(img)
        predictions = self._run_inference(inp_tensor)

        return predictions

class CLIPEngineDF40:
    INPUT_SIZE = (224, 224)
    _INPUT_MEAN = [0.5, 0.5, 0.5]
    _INPUT_STD = [0.5, 0.5, 0.5]

    def __init__(self, device_name: str = "cpu", use_triton=False):
        """
        Initialize the CLIPEngine with the specified device.

        Args:
            device_name (str, optional): The device to use for inference. Defaults to "cpu".
                Options are "cpu" for CPUExecutionProvider and "cuda" for CUDAExecutionProvider.

        This initializes an ONNX inference session with the model specified by the model path.
        If CUDA is not available or fails, it defaults to using the CPU.
        """
        model_path = Path(__file__).parent / "weights/clip_df40_large.onnx"
        self.use_triton = use_triton
        if use_triton:
            print(f"{self.__class__.__name__} use Triton")
            self.net = grpcclient.InferenceServerClient(
                url="grpc.aws-production.setoranku.com:443",
                ssl=True,
                root_certificates=None,  # Use system default certificates
                private_key=None,
                certificate_chain=None
            )
        else:    
            if device_name == "cpu":
                self.net = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
            else:
                try:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CUDAExecutionProvider"]
                    )
                except:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )
                    
        print(f"{self.__class__.__name__} net class: {self.net.__class__}")

    def _preproce_data(
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
        img = _normalize(img, self._INPUT_MEAN, self._INPUT_STD)
        img = np.expand_dims(img, axis=0)
        return img

    def _run_inference(self, inp_tensor: np.ndarray) -> float:
        """
        Run the inference.

        Args:
            inp_tensor (np.ndarray): The input tensor.

        Returns:
            float: Logits score.
        """
        if self.use_triton:
            inputs = []
            outputs = []
            inputs.append(grpcclient.InferInput("input", [1, 3, 224, 224], "FP32"))
            outputs.append(grpcclient.InferRequestedOutput("output"))
            inputs[0].set_data_from_numpy(inp_tensor)

        
            results = self.net.infer(
            model_name="clip-detector",
            inputs=inputs,
            outputs=outputs,
            )
            # Get the output arrays from the results
            predictions = results.as_numpy("output")
        else:
            predictions = self.net.run(output_names=None, input_feed={"input": inp_tensor})

        return predictions

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Logits score indicating the likelihood of the image being a deepfake.
        """

        inp_tensor: np.ndarray = self._preproce_data(img)
        predictions = self._run_inference(inp_tensor)

        return predictions

class CLIPEngineEffort:
    INPUT_SIZE = (224, 224)
    _INPUT_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _INPUT_STD = [0.26862954, 0.26130258, 0.27577711]
    
    def __init__(self, device_name: str = "cpu", use_triton=False):
        """
        Initialize the CLIPEngine with the specified device.

        Args:
            device_name (str, optional): The device to use for inference. Defaults to "cpu".
                Options are "cpu" for CPUExecutionProvider and "cuda" for CUDAExecutionProvider.

        This initializes an ONNX inference session with the model specified by the model path.
        If CUDA is not available or fails, it defaults to using the CPU.
        """
        model_path = Path(__file__).parent / "weights/effort_clip_L14_trainOn_FaceForensic.onnx"
        self.use_triton = use_triton
        if use_triton:
            print(f"{self.__class__.__name__} use Triton")
            self.net = grpcclient.InferenceServerClient(
                url="grpc.aws-production.setoranku.com:443",
                ssl=True,
                root_certificates=None,  # Use system default certificates
                private_key=None,
                certificate_chain=None
            )
        else:    
            if device_name == "cpu":
                self.net = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
            else:
                try:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CUDAExecutionProvider"]
                    )
                except:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )
                    
        print(f"{self.__class__.__name__} net class: {self.net.__class__}")

    def _preproce_data(
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
        img = _normalize(img, self._INPUT_MEAN, self._INPUT_STD)
        img = np.expand_dims(img, axis=0)
        return img

    def _run_inference(self, inp_tensor: np.ndarray) -> float:
        """
        Run the inference.

        Args:
            inp_tensor (np.ndarray): The input tensor.

        Returns:
            float: Logits score.
        """
        if self.use_triton:
            inputs = []
            outputs = []
            inputs.append(grpcclient.InferInput("input", [1, 3, 224, 224], "FP32"))
            outputs.append(grpcclient.InferRequestedOutput("output"))
            inputs[0].set_data_from_numpy(inp_tensor)

        
            results = self.net.infer(
            model_name="clip-detector",
            inputs=inputs,
            outputs=outputs,
            )
            # Get the output arrays from the results
            predictions = results.as_numpy("output")
        else:
            predictions = self.net.run(output_names=None, input_feed={"input": inp_tensor})

        return predictions

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Logits score indicating the likelihood of the image being a deepfake.
        """

        inp_tensor: np.ndarray = self._preproce_data(img)
        predictions = self._run_inference(inp_tensor)

        return predictions

class CLIPEngineEffort_VH:
    INPUT_SIZE = (224, 224)
    _INPUT_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _INPUT_STD = [0.26862954, 0.26130258, 0.27577711]
    
    def __init__(self, device_name: str = "cpu", use_triton=False):
        """
        Initialize the CLIPEngine with the specified device.

        Args:
            device_name (str, optional): The device to use for inference. Defaults to "cpu".
                Options are "cpu" for CPUExecutionProvider and "cuda" for CUDAExecutionProvider.

        This initializes an ONNX inference session with the model specified by the model path.
        If CUDA is not available or fails, it defaults to using the CPU.
        """
        model_path = Path(__file__).parent / "weights/effort_vh_1218.onnx"
        self.use_triton = use_triton
        if use_triton:
            print(f"{self.__class__.__name__} use Triton")
            self.net = grpcclient.InferenceServerClient(
                url="grpc.aws-production.setoranku.com:443",
                ssl=True,
                root_certificates=None,  # Use system default certificates
                private_key=None,
                certificate_chain=None
            )
        else:    
            if device_name == "cpu":
                self.net = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
            else:
                try:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CUDAExecutionProvider"]
                    )
                except:
                    self.net = ort.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )
                    
        print(f"{self.__class__.__name__} net class: {self.net.__class__}")

    def _preproce_data(
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
        img = _normalize(img, self._INPUT_MEAN, self._INPUT_STD)
        img = np.expand_dims(img, axis=0)
        return img

    def _run_inference(self, inp_tensor: np.ndarray) -> float:
        """
        Run the inference.

        Args:
            inp_tensor (np.ndarray): The input tensor.

        Returns:
            float: Logits score.
        """
        if self.use_triton:
            inputs = []
            outputs = []
            inputs.append(grpcclient.InferInput("input", [1, 3, 224, 224], "FP32"))
            outputs.append(grpcclient.InferRequestedOutput("output"))
            inputs[0].set_data_from_numpy(inp_tensor)

        
            results = self.net.infer(
            model_name="clip-detector",
            inputs=inputs,
            outputs=outputs,
            )
            # Get the output arrays from the results
            predictions = results.as_numpy("output")
        else:
            predictions = self.net.run(output_names=None, input_feed={"input": inp_tensor})

        return predictions

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Logits score indicating the likelihood of the image being a deepfake.
        """

        inp_tensor: np.ndarray = self._preproce_data(img)
        predictions = self._run_inference(inp_tensor)

        return predictions
