import random

import torchvision.transforms as transforms
from PIL import Image, ImageFilter

class RandomBrightness:
    def __init__(self, brightness=(0.4, 1.6), p=0.5):
        self.brightness_augmentation = transforms.ColorJitter(brightness=brightness)

        self.p = p

    def _should_apply(self):
        if random.random() < self.p:
            return True

        return False

    def __call__(self, x):
        if not self._should_apply():
            return x

        augmented_image = self.brightness_augmentation(x)

        return augmented_image


class SimulateLowQuality(object):
    def __init__(self, p=0.5):
        self.p = p
        self.transform = transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.GaussianBlur(kernel_size=5),  # Simulate blur
                        transforms.RandomAffine(
                            degrees=0, translate=(0.1, 0.1)
                        ),  # Simulate slight translation
                    ]
                ),
                transforms.RandomChoice(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        ),  # Adjust color
                        transforms.RandomGrayscale(
                            p=0.2,
                        ),  # Add random grayscale
                    ]
                ),
            ]
        )

    def simulate_lower_bit_depth(self, img):
        quantized_img = img.quantize(colors=16)
        return quantized_img

    def use_adaptive_palette(self, img):
        img = img.convert("P", palette=Image.ADAPTIVE, colors=16)
        return img

    # Randomly choose one of the transformations
    def random_choice_transform(self, img):
        choices = [self.simulate_lower_bit_depth, self.use_adaptive_palette]
        chosen_transform = random.choice(choices)
        return chosen_transform(img)

    def _should_apply(self):
        if random.random() < self.p:
            return True

        return False

    def __call__(self, x):
        if not self._should_apply():
            return x
        x = self.transform(x)
        x = self.random_choice_transform(x)
        if x.mode == "L" or x.mode == "P":
            # Convert the single-channel image to a three-channel (RGB) image
            x = x.convert("RGB")

        return x


class RandomDownUpSampler:
    def __init__(self, interpolation_methods=None, downsampling_range=(0.3,0.8), p=0.5):
        if interpolation_methods is None:
            self.interpolation_methods = [
                Image.NEAREST,
                Image.BILINEAR,
                Image.BICUBIC,
                Image.LANCZOS,
            ]
        else:
            self.interpolation_methods = interpolation_methods

        self.p = p
        self.downsampling_range = downsampling_range

    def _should_apply(self):
        if random.random() < self.p:
            return True

        return False

    def __call__(self, x):
        if not self._should_apply():
            return x

        init_size = x.size

        downsampling_interpolation = random.choice(self.interpolation_methods)
        upsampling_interpolation = random.choice(self.interpolation_methods)
    
        downsampling_factor = random.uniform(*self.downsampling_range)

        downsampled_image = x.resize(
            (
                int(x.width * downsampling_factor),
                int(x.height * downsampling_factor),
            ),
            downsampling_interpolation,
        )

        upsampled_image = downsampled_image.resize(init_size, upsampling_interpolation)

        return upsampled_image

class SimCLRGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x