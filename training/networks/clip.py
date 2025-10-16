import timm
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet

# from pytorch_pretrained_vit import ViT
from torch.nn import init
from metrics.registry import BACKBONE
#from timm.models.repvit import NormLinear
from transformers import AutoProcessor, CLIPModel

def get_last_linear(module: nn.Module):
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    return None

# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(
            m.weight.data, a=0, mode="fan_in"
        )  # For old pytorch, you may use kaiming_normal.
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        init.constant_(m.bias.data, 0.0)

    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


@BACKBONE.register_module(module_name="clip")
class CLIPVisual(nn.Module):
    def __new__(cls, config_clip):
        model_name = config_clip["model_name"]
        bb = get_clip_visual(model_name)[1]
        bb.n_features = get_last_linear(bb.encoder.layers[-1]).out_features
        
        return bb
    
    #def __init__(self, clip_config):
    #    super(CLIPVisual, self).__init__()
    #    self.net = get_clip_visual()[1]
    #    return 
    
    def features(self, x):
        x = self.forward(x)
        return x 
    
    def forward(self, x):
        x = self.features(x)
        return x


@BACKBONE.register_module(module_name="clip_effort")
class CLIPEffort(nn.Module):
    def __new__(cls, config_clip):
        model_name = config_clip["model_name"]
        bb = get_clip_visual(model_name)[1]
        bb.n_features = get_last_linear(bb.encoder.layers[-1]).out_features
        
        return bb
    
    #def __init__(self, clip_config):
    #    super(CLIPVisual, self).__init__()
    #    self.net = get_clip_visual()[1]
    #    return 
    
    def features(self, x):
        x = self.forward(x)
        return x 
    
    def forward(self, x):
        x = self.features(x)
        return x
      
def get_clip_visual(model_name="openai/clip-vit-base-patch16"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model

#@BACKBONE.register_module(module_name=clipeffort)