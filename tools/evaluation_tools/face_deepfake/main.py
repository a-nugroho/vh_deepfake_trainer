import numpy as np
import time
from .engine_clip_onnx import CLIPEngine, CLIPEngineDF40, CLIPEngineEffort, CLIPEngineEffort_VH
from .engine_onnx import DeepfakeEngine, DeepfakeEngineTriton


def softmax(x):
    """
    Compute the softmax of a vector.

    The softmax function is a type of normalization that scales the elements
    of the input vector to be between 0 and 1, while maintaining the relative
    proportions of the values. It is commonly used in machine learning models
    to convert raw prediction values into probabilities.

    Args:
        x (array-like): Input array or list of values.

    Returns:
        np.ndarray: An array of the same shape as `x`, with values transformed
        to represent probabilities that sum to 1.
    """

    return np.exp(x) / np.exp(x).sum()


class DeepfakeDetection:
    """Face deepfake checking using deep network to predict deepfake score."""

    MAIN_SCORE_THRESHOLD = DeepfakeEngine.DEFAULT_THRESHOLD

    def __init__(self, device_name: str = "cpu"):
        """
        Initialize the DeepfakeDetection with the specified device.

        This constructor sets up two engines for deepfake detection using the given device.
        The engines are `DeepfakeEngine` and `CLIPEngine`, both of which will utilize
        the specified device for inference.

        Args:
            device_name (str, optional): The device to run the engines on.
                Defaults to "cpu". Can be "cuda" for GPU inference.
        """

        self.engine_1 = DeepfakeEngine(device_name=device_name)
        self.engine_2 = CLIPEngine(device_name=device_name)

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score. Internally, two engines are used
        to predict the score and the results are ensembled.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Deepfake score.
        """
        predictions_1 = self.engine_1.predict(img)[0]
        predictions_2 = self.engine_2.predict(img)[0]
        ensemble_logits = (predictions_1 + predictions_2) / 2

        return softmax(ensemble_logits)[0][1].item()

    @staticmethod
    def classify_predictions(main_score: float) -> bool:
        """Decide whether the score indicates deepfake or none.

        Args:
            main_score (float): The deepfake score.

        Returns:
            bool: Deepfake decision of either deepfake (`True`) or no deepfake (`False`).
        """
        return main_score > DeepfakeDetection.MAIN_SCORE_THRESHOLD


class DeepfakeDetectionDF40:
    """Face deepfake checking using deep network to predict deepfake score."""

    MAIN_SCORE_THRESHOLD = DeepfakeEngine.DEFAULT_THRESHOLD

    def __init__(self, device_name: str = "cpu"):
        """
        Initialize the DeepfakeDetection with the specified device.

        This constructor sets up two engines for deepfake detection using the given device.
        The engines are `DeepfakeEngine` and `CLIPEngine`, both of which will utilize
        the specified device for inference.

        Args:
            device_name (str, optional): The device to run the engines on.
                Defaults to "cpu". Can be "cuda" for GPU inference.
        """

        self.engine_1 = CLIPEngineDF40(device_name=device_name)

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score. Internally, two engines are used
        to predict the score and the results are ensembled.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Deepfake score.
        """
        predictions_1 = self.engine_1.predict(img)[0]
        ensemble_logits = predictions_1

        return softmax(ensemble_logits)[0][1].item()

    @staticmethod
    def classify_predictions(main_score: float) -> bool:
        """Decide whether the score indicates deepfake or none.

        Args:
            main_score (float): The deepfake score.

        Returns:
            bool: Deepfake decision of either deepfake (`True`) or no deepfake (`False`).
        """
        return main_score > DeepfakeDetection.MAIN_SCORE_THRESHOLD

class DeepfakeDetectionEffort:
    """Face deepfake checking using deep network to predict deepfake score."""

    MAIN_SCORE_THRESHOLD = DeepfakeEngine.DEFAULT_THRESHOLD

    def __init__(self, device_name: str = "cpu"):
        """
        Initialize the DeepfakeDetection with the specified device.

        This constructor sets up two engines for deepfake detection using the given device.
        The engines are `DeepfakeEngine` and `CLIPEngine`, both of which will utilize
        the specified device for inference.

        Args:
            device_name (str, optional): The device to run the engines on.
                Defaults to "cpu". Can be "cuda" for GPU inference.
        """

        self.engine_1 = CLIPEngineEffort(device_name=device_name)

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score. Internally, two engines are used
        to predict the score and the results are ensembled.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Deepfake score.
        """
        predictions_1 = self.engine_1.predict(img)[0]
        ensemble_logits = predictions_1

        return softmax(ensemble_logits)[0][1].item()

    @staticmethod
    def classify_predictions(main_score: float) -> bool:
        """Decide whether the score indicates deepfake or none.

        Args:
            main_score (float): The deepfake score.

        Returns:
            bool: Deepfake decision of either deepfake (`True`) or no deepfake (`False`).
        """
        return main_score > DeepfakeDetection.MAIN_SCORE_THRESHOLD

class DeepfakeDetectionEffort_VH:
    """Face deepfake checking using deep network to predict deepfake score."""

    MAIN_SCORE_THRESHOLD = DeepfakeEngine.DEFAULT_THRESHOLD

    def __init__(self, device_name: str = "cpu"):
        """
        Initialize the DeepfakeDetection with the specified device.

        This constructor sets up two engines for deepfake detection using the given device.
        The engines are `DeepfakeEngine` and `CLIPEngine`, both of which will utilize
        the specified device for inference.

        Args:
            device_name (str, optional): The device to run the engines on.
                Defaults to "cpu". Can be "cuda" for GPU inference.
        """

        self.engine_1 = CLIPEngineEffort_VH(device_name=device_name)

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score. Internally, two engines are used
        to predict the score and the results are ensembled.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Deepfake score.
        """
        predictions_1 = self.engine_1.predict(img)[0]
        ensemble_logits = predictions_1

        return softmax(ensemble_logits)[0][1].item()

    @staticmethod
    def classify_predictions(main_score: float) -> bool:
        """Decide whether the score indicates deepfake or none.

        Args:
            main_score (float): The deepfake score.

        Returns:
            bool: Deepfake decision of either deepfake (`True`) or no deepfake (`False`).
        """
        return main_score > DeepfakeDetection.MAIN_SCORE_THRESHOLD


class DeepfakeDetectionTriton:
    """Face deepfake checking using deep network to predict deepfake score."""

    MAIN_SCORE_THRESHOLD = DeepfakeEngine.DEFAULT_THRESHOLD

    def __init__(self, device_name: str = "cpu",convnext_triton=False,clip_triton=False):
        """
        Initialize the DeepfakeDetection with the specified device.

        This constructor sets up two engines for deepfake detection using the given device.
        The engines are `DeepfakeEngine` and `CLIPEngine`, both of which will utilize
        the specified device for inference.

        Args:
            device_name (str, optional): The device to run the engines on.
                Defaults to "cpu". Can be "cuda" for GPU inference.
        """

        self.engine_1 = DeepfakeEngine(device_name=device_name,use_triton=convnext_triton)
        self.engine_2 = CLIPEngine(device_name=device_name,use_triton=clip_triton)

    def predict(self, img: np.ndarray) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score. Internally, two engines are used
        to predict the score and the results are ensembled.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Deepfake score.
        """
        predictions_1 = self.engine_1.predict(img)[0]
        predictions_2 = self.engine_2.predict(img)[0]
        ensemble_logits = (predictions_1 + predictions_2) / 2
        if ensemble_logits.ndim==1:
            return softmax(ensemble_logits)[1].item()
        else:
            return softmax(ensemble_logits)[0][1].item()

    @staticmethod
    def classify_predictions(main_score: float) -> bool:
        """Decide whether the score indicates deepfake or none.

        Args:
            main_score (float): The deepfake score.

        Returns:
            bool: Deepfake decision of either deepfake (`True`) or no deepfake (`False`).
        """
        return main_score > DeepfakeDetection.MAIN_SCORE_THRESHOLD

class DeepfakeDetectionCustom:
    """Face deepfake checking using deep network to predict deepfake score."""

    MAIN_SCORE_THRESHOLD = DeepfakeEngine.DEFAULT_THRESHOLD

    def __init__(self, device_name: str = "cpu", list_dict_engine={}):
        """
        Initialize the DeepfakeDetection with the specified device.

        This constructor sets up two engines for deepfake detection using the given device.
        The engines are `DeepfakeEngine` and `CLIPEngine`, both of which will utilize
        the specified device for inference.

        Args:
            device_name (str, optional): The device to run the engines on.
                Defaults to "cpu". Can be "cuda" for GPU inference.
        """
        #self.engine_1 = DeepfakeEngine(device_name=device_name,use_triton=convnext_triton)
        #self.engine_2 = CLIPEngine(device_name=device_name,use_triton=clip_triton)

        self.engines=[]
        for dict_now in list_dict_engine:
            model_now = dict_now['engine_class']
            del dict_now['engine_class']
            dict_now["device_name"]=device_name
            engine_now = model_now(**dict_now)
            self.engines.append(engine_now)
    
    def predict(self, img: np.ndarray, return_all=False, return_time=False) -> float:
        """
        Predict deepfake score from the given image.

        This method takes an image (as a BGR-ordered NumPy array) and
        predicts the deepfake score. Internally, two engines are used
        to predict the score and the results are ensembled.

        Args:
            img (np.ndarray): A BGR-ordered image.

        Returns:
            float: Deepfake score.
        """
        pred_list = []
        time_list = []
        for eng_now in self.engines:
            start_time = time.time()
            pred_list.append(eng_now.predict(img)[0])
            time_ms = (time.time()-start_time)*1000
            time_list.append(time_ms)
        if return_all:
            pred_all = [i[0] for i in pred_list]
        ensemble_logits = np.array(pred_list)
        ensemble_logits = np.sum(pred_list,axis=0)/len(self.engines)
        return_list = [softmax(ensemble_logits)[0][1].item()]
        if return_all: 
            return_list.append(pred_all)
        if return_time:
            return_list.append(time_list)
            #return softmax(ensemble_logits)[0][1].item(), pred_all
        
        return return_list
        
        #else:
        #    return softmax(ensemble_logits)[0][1].item()

    @staticmethod
    def classify_predictions(main_score: float) -> bool:
        """Decide whether the score indicates deepfake or none.

        Args:
            main_score (float): The deepfake score.

        Returns:
            bool: Deepfake decision of either deepfake (`True`) or no deepfake (`False`).
        """
        return main_score > DeepfakeDetection.MAIN_SCORE_THRESHOLD
