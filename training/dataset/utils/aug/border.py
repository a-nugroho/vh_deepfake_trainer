import random
from enum import IntEnum
from typing import Union, Tuple, Sequence

import cv2
import numpy as np
from PIL import Image


class BorderLocation(IntEnum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

from PIL import Image
import random

class RandomBlackBorderFixedSizeSquare:
    def __init__(self, max_border_ratio=0.3, p=0.5):
        """
        max_border_ratio: max proportion (0 to 1) of P that a single border can take
        p: probability to apply the transform
        """
        self.max_border_ratio = max_border_ratio
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        w, h = img.size
        assert w == h, "This transform only works on square images (P x P)."

        P = w
        # Random borders (max e.g. 30% of P)
        top = random.randint(0, int(self.max_border_ratio * P))
        bottom = random.randint(0, int(self.max_border_ratio * P))
        left = random.randint(0, int(self.max_border_ratio * P))
        right = random.randint(0, int(self.max_border_ratio * P))

        # Crop the original image inward to fit the area not covered by borders
        cropped = img.crop((left, top, P - right, P - bottom))

        # Resize the cropped part to fit back in P x P
        resized = cropped.resize((P - left - right, P - top - bottom))

        # Create black background and paste the resized image
        result = Image.new("RGB", (P, P), color=(0, 0, 0))
        result.paste(resized, (left, top))

        return result

class RandomBorder:
    def __init__(
        self,
        border_amount: Union[Tuple[float], float] = (0.01, 0.15),
        border_type: int = cv2.BORDER_CONSTANT,
        p: float = 0.5,
    ) -> None:
        # """
        # Randomly replace one of image sides with border.

        # Args:
        #     border_amount (Union[Tuple[float], float]): The size of border in
        #         relation to image width or height. If 2-tuple of float
        #         is passed, the size will be randomized with min and max values.
        #         If single float is passed, the border size will be the same.
        #         Defaults to (0.1, 0.3).
        #     border_type (int): The type of border to use.
        #         Uses OpenCV border enum. Defaults to cv2.BORDER_CONSTANT.
        #     p (float): The probability of applying the border. Defaults to 0.5.
        # """
        
        if isinstance(border_amount, float):
            border_amount = (border_amount, border_amount)
        elif not isinstance(border_amount, Sequence):
            raise TypeError(
                "border_amount must be either float or 2-item sequence, got "
                f"{type(border_amount)}"
            )

        if len(border_amount) != 2:
            raise ValueError(
                "border_amount must be 2-item sequence, got {}-item sequence".format(
                    len(border_amount)
                )
            )

        if not isinstance(p, float):
            raise TypeError("p must be a float")
        elif p < 0 or p > 1:
            raise TypeError("p must be between 0 and 1 inclusive")

        self.border_amount = border_amount
        self.p = p
        self.border_type = border_type

    def _should_apply(self):
        if random.random() < self.p:
            return True

        return False

    def apply(self, img: Union[np.ndarray, Image.Image]):
        if not self._should_apply():
            return img

        ret_as_pil = False
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            ret_as_pil = True

        ORIG_H, ORIG_W, _ = img.shape
        border_loc = random.choice(list(BorderLocation))

        amnt = random.uniform(*self.border_amount)
        amnt_x = int(ORIG_W * amnt)
        amnt_y = int(ORIG_H * amnt)

        btop, bbottom, bleft, bright = (0, 0, 0, 0)
        if border_loc == BorderLocation.TOP:
            btop = amnt_y
            img = img[amnt_y:, :]
        elif border_loc == BorderLocation.LEFT:
            bleft = amnt_x
            img = img[:, amnt_x:]
        elif border_loc == BorderLocation.RIGHT:
            bright = amnt_x
            img = img[:, : ORIG_W - amnt_x]
        else:
            bbottom = amnt_y
            img = img[: ORIG_H - amnt_y, :]

        img = cv2.copyMakeBorder(
            img, btop, bbottom, bleft, bright, borderType=self.border_type
        )

        if ret_as_pil:
            return Image.fromarray(img)

        return img

    def __call__(self, img: Union[np.ndarray, Image.Image]):
        return self.apply(img)
