import numpy as np

from .engine_clip_onnx import CLIPEngine
from .engine_onnx import DeepfakeEngine, SKLearnEnsembler

ENGINE_REF = {"clip_cip":{'engine_class':DeepfakeEngine,
        'input_size':(224, 224),
        'input_mean':(0.5, 0.5, 0.5),
        'input_std':(0.5, 0.5, 0.5),
        'model_path':'clip_detector_optimized.onnx',
        'logit_offset':(0, 3.0),
        'logit_temp':1.25
    },
    "clip_df40pre":{'engine_class':DeepfakeEngine,
        'input_size':(224, 224),
        'input_mean':(0.5, 0.5, 0.5),
        'input_std':(0.5, 0.5, 0.5),
        'model_path':'clip_large_df40_allff.onnx',
        'logit_offset':(0, -1.0),
        'logit_temp':1.25
    },
    "clip_eff_vh":{'engine_class':DeepfakeEngine,
        'input_size':(224, 224),
        'input_mean':(0.5, 0.5, 0.5),
        'input_std':(0.5, 0.5, 0.5),
        'model_path':'effort_vh_2026-01-04.onnx',
        'logit_offset':(0, 0.5),
        'logit_temp':1.0
    }
}
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

    MAIN_SCORE_THRESHOLD = 0.6

    def __init__(self, device_name: str = "cpu", model_weights=None, main_score_threshold = None):
        """
        Initialize the DeepfakeDetection with the specified device.

        This constructor sets up two engines for deepfake detection using the given device.
        The engines are `DeepfakeEngine` and `CLIPEngine`, both of which will utilize
        the specified device for inference.

        Args:
            device_name (str, optional): The device to run the engines on.
                Defaults to "cpu". Can be "cuda" for GPU inference.
        """
        list_dict_engine = ENGINE_REF
        self.engines=[]
        self.model_weights=model_weights
        for dict_now in list_dict_engine:
            model_now = dict_now['engine_class']
            del dict_now['engine_class']
            dict_now["device_name"]=device_name
            engine_now = model_now(**dict_now)
            self.engines.append(engine_now)

        if main_score_threshold is not None:
            DeepfakeDetection.MAIN_SCORE_THRESHOLD = main_score_threshold

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
        pred_list = []
        time_list = []
        for eng_now in self.engines:
            pred_list.append(eng_now.predict(img)[0])
        
        if self.model_weights:
            ensemble_logits = [pred_list[i]*self.model_weights[i] for i in range(len(self.engines))]
            ensemble_logits = np.array(pred_list)
            ensemble_logits = np.sum(pred_list,axis=0)
        else:
            ensemble_logits = np.array(pred_list)
            ensemble_logits = np.sum(pred_list,axis=0)/len(self.engines)
       
        ens_score = softmax(ensemble_logits)[0][1].item()
        return ens_score

    @staticmethod
    def classify_predictions(main_score: float) -> bool:
        """Decide whether the score indicates deepfake or none.

        Args:
            main_score (float): The deepfake score.

        Returns:
            bool: Deepfake decision of either deepfake (`True`) or no deepfake (`False`).
        """
        return main_score > DeepfakeDetection.MAIN_SCORE_THRESHOLD
