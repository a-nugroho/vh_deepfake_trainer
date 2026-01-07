import torch
import torch.nn as nn
import yaml
from transformers import AutoProcessor, CLIPModel


class CLIPDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = self.build_backbone()
        self.head = nn.Linear(768, 2)

    def build_backbone(self):
        # prepare the backbone
        _, backbone = get_clip_visual(model_name="openai/clip-vit-base-patch16")
        return backbone

    def features(self, image: torch.tensor) -> torch.tensor:
        feat = self.backbone(image)["pooler_output"]
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def forward(self, image: torch.tensor) -> torch.tensor:
        # get the features by backbone
        features = self.features(image)

        # get the prediction by classifier
        pred = self.classifier(features)

        return pred


def get_clip_visual(model_name="openai/clip-vit-base-patch16"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model
