#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : model_dicts
# @Date : 2022-09-03-13-46
# @Project : classification_models_pytorch
# @Author : seungmin


from torchvision.models import *


resnet_dict = {"resnet18": resnet18,
               "resnet34": resnet34,
               "resnet50": resnet50}

visionTransformer_dict = {"vit_b_16": vit_b_16,
                          "vit_b_32": vit_b_32,
                          "vit_l_16": vit_l_16,
                          "vit_l_32": vit_l_32,
                          "vit_h_14": vit_h_14}

swinTransformer_dict = {"swin_t": swin_t,
                        "swin_s": swin_s,
                        "swin_b": swin_b}

weights_dict = {"imagenet1k_v1": "IMAGENET1K_V1",
                "imagenet1k_v2": "IMAGENET1K_V2",
                "v1": "IMAGENET1K_V1",
                "v2": "IMAGENET1K_V2",
                "default": "DEFAULT",
                "random": None}