import torch
import torch.nn as nn
from .model_dicts import *


class ResNet(nn.Module):
    def __init__(self, feature_ext, classes, weight):
        super(ResNet, self).__init__()
        self.resnet_dict = resnet_dict
        self.weights_dict = weights_dict

        resnet = self._get_submodel(feature_ext, weight)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, classes)

    def _get_submodel(self, feature_extractor, pretrained_weight):
        try:
            model = self.resnet_dict[feature_extractor]
            weights = self.weights_dict[pretrained_weight]
            submodel = model(weights=weights)
            print("Feature extractor:", feature_extractor)
            return submodel
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet~, resnext~ or wide_resnet~.")

    def forward(self, x):
        h = self.features(x)
        h = self.linear(h.squeeze())
        return h


class MobileNet(nn.Module):
    def __init__(self, feature_ext, classes, weight):
        super(MobileNet, self).__init__()
        self.mobilenet_dict = mobilenet_dict
        self.weights_dict = weights_dict

        self.mobilenet = self._get_submodel(feature_ext, weight)
        self.mobilenet.classifier[-1] = nn.Linear(self.mobilenet.classifier[-1].in_features, classes)

    def _get_submodel(self, feature_extractor, pretrained_weight):
        try:
            model = self.mobilenet_dict[feature_extractor]
            weights = self.weights_dict[pretrained_weight]
            submodel = model(weights=weights)
            print("Feature extractor:", feature_extractor)
            return submodel
        except:
            raise ("Invalid model name. Check the config file and pass one of: mobilenet_v2, mobilenet_v3_small or mobilenet_v3_large.")

    def forward(self, x):
        h = self.mobilenet(x)
        return h


class EfficientNet(nn.Module):
    def __init__(self, feature_ext, classes, weight):
        super(EfficientNet, self).__init__()
        self.efficientnet_dict = efficientnet_dict
        self.weights_dict = weights_dict

        self.efficientnet = self._get_submodel(feature_ext, weight)
        self.efficientnet.classifier[-1] = nn.Linear(self.efficientnet.classifier[-1].in_features, classes)

    def _get_submodel(self, feature_extractor, pretrained_weight):
        try:
            model = self.efficientnet_dict[feature_extractor]
            weights = self.weights_dict[pretrained_weight]
            submodel = model(weights=weights)
            print("Feature extractor:", feature_extractor)
            return submodel
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet~, resnext~ or wide_resnet~.")

    def forward(self, x):
        h = self.efficientnet(x)
        return h


class VisionTransformer(nn.Module):
    def __init__(self, feature_ext, classes, weight):
        super(VisionTransformer, self).__init__()
        self.visionTransformer_dict = visionTransformer_dict
        self.weights_dict = weights_dict

        self.vit = self._get_submodel(feature_ext, weight)
        self.vit.heads[-1] = nn.Linear(self.vit.heads[-1].in_features, classes)

    def _get_submodel(self, feature_extractor, pretrained_weight):
        try:
            model = self.visionTransformer_dict[feature_extractor]
            weights = self.weights_dict[pretrained_weight]
            submodel = model(weights=weights)
            print("Feature extractor:", feature_extractor)
            return submodel
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet~, resnext~ or wide_resnet~.")

    def forward(self, x):
        h = self.vit(x)
        return h


class SwinTransformer(nn.Module):
    def __init__(self, feature_ext, classes, weight):
        super(SwinTransformer, self).__init__()
        self.swinTransformer_dict = swinTransformer_dict
        self.weights_dict = weights_dict

        self.swint = self._get_submodel(feature_ext, weight)
        self.swint.head = nn.Linear(self.swint.head.in_features, classes)

    def _get_submodel(self, feature_extractor, pretrained_weight):
        try:
            model = self.swinTransformer_dict[feature_extractor]
            weights = self.weights_dict[pretrained_weight]
            submodel = model(weights=weights)
            print("Feature extractor:", feature_extractor)
            return submodel
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet~, resnext~ or wide_resnet~.")

    def forward(self, x):
        h = self.swint(x)
        return h



if __name__ == "__main__":
    #model = VisionTransformer("vit_b_16", 2, "random")
    #model = SwinTransformer("swin_b", 7, "random")
    #model = ResNet("resnet18", 4, "random")
    #model = MobileNet("mobilenet_v3_large", 4, "random")
    model = EfficientNet("efficientnet_b7", 13, "random")

    dummy = torch.rand(4, 3, 1024, 1024)
    pred = model(dummy)
    print(pred)
    #
    #print(model)