import torch.nn as nn
from model_dicts import *


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


class VisionTransformer(nn.Module):
    def __init__(self, feature_ext, classes, weight):
        super(VisionTransformer, self).__init__()
        self.visionTransformer_dict = visionTransformer_dict
        self.weights_dict = weights_dict

        vit = self._get_submodel(feature_ext, weight)
        self.features = nn.Sequential(*list(vit.children())[:-1])
        self.linear = nn.Linear(vit.head[1].in_features, classes)

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
        h = self.features(x)
        ##h = self.linear(h.squeeze())
        return h



class MobileNet(nn.Module):
    def __init__(self, feature_ext, classes, pretrain):
        super(MobileNet, self).__init__()
        self.mobilenet_dict = {'mobilenet_v2' : models.mobilenet_v2(pretrained=pretrain),
                               'mobilenet_v3_small' : models.mobilenet_v3_small(pretrained=pretrain),
                               'mobilenet_v3_large' : models.mobilenet_v3_large(pretrained=pretrain)}

        self.mobilenet = self._get_submodel(feature_ext)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, classes)

    def _get_submodel(self, feature_extractor):
        try:
            model = self.mobilenet_dict[feature_extractor]
            print("Feature extractor:", feature_extractor)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: mobilenet_v2, mobilenet_v3_small or mobilenet_v3_large.")

    def forward(self, x):
        h = self.mobilenet(x)
        return h

if __name__ == "__main__":
    model = VisionTransformer("vit_b_16", 4, "random")
    #model = ResNet("resnet18", 4, "random")
    print(model)