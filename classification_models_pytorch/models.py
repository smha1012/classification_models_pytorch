import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, feature_ext, classes, pretrain):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrain),
                            "resnet34": models.resnet34(pretrained=pretrain),
                            "resnet50": models.resnet50(pretrained=pretrain),
                            "resnet101": models.resnet101(pretrained=pretrain),
                            "resnet152": models.resnet152(pretrained=pretrain),
                            "resnext50_32x4d": models.resnext50_32x4d(pretrained=pretrain),
                            "resnext101_32x8d": models.resnext101_32x8d(pretrained=pretrain),
                            "wide_resnet50_2": models.wide_resnet50_2(pretrained=pretrain),
                            "wide_resnet101_2": models.wide_resnet101_2(pretrained=pretrain)}

        resnet = self._get_submodel(feature_ext)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, classes)

    def _get_submodel(self, feature_extractor):
        try:
            model = self.resnet_dict[feature_extractor]
            print("Feature extractor:", feature_extractor)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet~, resnext~ or wide_resnet~.")

    def forward(self, x):
        h = self.features(x)
        h = self.linear(h.squeeze())
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