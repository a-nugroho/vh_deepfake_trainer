import timm
import torch
import torch.nn as nn
import torchvision

# from pytorch_pretrained_vit import ViT
from torch.nn import init
from metrics.registry import BACKBONE
#from timm.models.repvit import NormLinear


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


# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel // 3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat(
                [_conv_stem_weight_new, _conv_stem_weight], axis=1
            )

    return torch.nn.Parameter(_conv_stem_weight_new)


def get_convnext(
    model_name="convnext_xlarge_384_in22ft1k",
    pretrained=True,
    num_classes=2,
    return_features=False,
):
    """
    :param model_name: convnext_xlarge_384_in22ft1k
    :param pretrained: Whether to load pretrained weights
    :param num_classes: The number of output classes for classification
    :param return_features: Whether to return the features before the classifier
    :return: The ConvNeXt model (with or without final classification layer)
    """
    net = timm.create_model(model_name, pretrained=pretrained)

    if return_features:
        # Remove the classifier to return features
        net.head = nn.Identity()

    else:
        # Modify the classifier to fit the number of classes
        n_features = net.head.fc.in_features
        net.head.fc = nn.Linear(n_features, num_classes)

    return net


@BACKBONE.register_module(module_name="convnext")
class ConvNext(nn.Module):
    def __new__(cls, config_convnext):
        model_name = config_convnext["model_name"]
        pretrained = config_convnext["pretrained"]
        bb = get_convnext(model_name, pretrained=pretrained)
        bb.n_features = get_last_linear(bb.stages[-1].blocks[-1]).out_features
        bb.head = nn.Identity()  # remove head
        return bb
    
    def __init__(self, config_convnext):
        super(ConvNext, self).__init__()
        """ Constructor
        Args:
            resnet_config: configuration file with the dict format
        """
        self.num_classes = config_convnext["num_classes"]
        model_name = config_convnext["model_name"]
        pretrained = config_convnext["pretrained"]
        #self.net = get_convnext(model_name, pretrained=pretrained)
        #self.n_features = get_last_linear(self.net.stages[-1].blocks[-1]).out_features

    def features(self, x, requires_feat=False):
        self.model = self.stages
        feat = []
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return (x, feat) if requires_feat else x

    def classifier(self, features):
        x = self.forward_head(features)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out