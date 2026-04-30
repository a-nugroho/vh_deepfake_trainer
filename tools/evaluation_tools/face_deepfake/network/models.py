import torch
import torch.nn as nn
import timm


def get_convnext(model_name='connect_xlarge_384_in22ft1k', pretrained=True, num_classes=2):
    """
    :param model_name: convnext_xlarge_384_in22ft1k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    return net


if __name__ == '__main__':
    model, image_size = get_convnext(model_name='connect_xlarge_384_in22ft1k', num_classes=2), 384
    # print(model.model.image_size)
    print(model)

    model = model.to(torch.device('cpu'))
    from torchsummary import summary

    input_s = (3, image_size, image_size)
    print(summary(model, input_s, device='cpu'))

    # print(model._modules.items())
    # print(model)

    pass
