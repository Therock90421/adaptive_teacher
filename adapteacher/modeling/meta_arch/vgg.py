# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn
import copy
import torch
from typing import Union, List, Dict, Any, cast
from detectron2.modeling.backbone import (
    ResNet,
    Backbone,
    build_resnet_backbone,
    BACKBONE_REGISTRY
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7



def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class vgg_backbone(Backbone):
    """
    Backbone (bottom-up) for FBNet.

    Hierarchy:
        trunk0:
            xif0_0
            xif0_1
            ...
        trunk1:
            xif1_0
            xif1_1
            ...
        ...

    Output features:
        The outputs from each "stage", i.e. trunkX.
    """

    def __init__(self, cfg):
        super().__init__()

        self.vgg = make_layers(cfgs['vgg16'],batch_norm=True)

        self._initialize_weights()
        # self.stage_names_index = {'vgg1':3, 'vgg2':8 , 'vgg3':15, 'vgg4':22, 'vgg5':29}
        _out_feature_channels = [64, 128, 256, 512, 512]
        _out_feature_strides = [2, 4, 8, 16, 32]
        # stages, shape_specs = build_fbnet(
        #     cfg,
        #     name="trunk",
        #     in_channels=cfg.MODEL.FBNET_V2.STEM_IN_CHANNELS
        # )

        # nn.Sequential(*list(self.vgg.features._modules.values())[:14])

        self.stages = [nn.Sequential(*list(self.vgg._modules.values())[0:7]),\
                    nn.Sequential(*list(self.vgg._modules.values())[7:14]),\
                    nn.Sequential(*list(self.vgg._modules.values())[14:24]),\
                    nn.Sequential(*list(self.vgg._modules.values())[24:34]),\
                    nn.Sequential(*list(self.vgg._modules.values())[34:]),]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "vgg{}".format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

        diff_in_channels = [256, 512, 512]
        diff_out_channels = [256, 512, 512]
        self.inv_attention = []
        self.spc = []
        self.spc_attention = []
        self.channel_attention = []
        self.spatial_attention = []
        for i, plane in enumerate(diff_in_channels):
            #domain invariant 特征的权重系数，1*1+GN
            inv_attention_layer = self.make_attention_layers(plane, plane)
            inv_attention_layer_name = f'inv_attention_layer{i + 2}'
            self.add_module(inv_attention_layer_name, inv_attention_layer)
            self.inv_attention.append(inv_attention_layer_name)

            #domain specific 特征,将diff后的特征做一次3*3conv + relu
            spc_layer = self.make_diff_layers(plane, plane)
            spc_layer_name = f'spc_layer{i + 2}'
            self.add_module(spc_layer_name, spc_layer)
            self.spc.append(spc_layer_name)

            #domain specific 特征的权重系数，1*1+GN
            spc_attention_layer = self.make_attention_layers(plane, plane)
            spc_attention_layer_name = f'spc_attention_layer{i + 2}'
            self.add_module(spc_attention_layer_name, spc_attention_layer)
            self.spc_attention.append(spc_attention_layer_name)

            #channel attention
            channel_attention_layer = self.make_channel_attention_layers(plane)
            channel_attention_layer_name = f'channel_attention_layer{i + 2}'
            self.add_module(channel_attention_layer_name, channel_attention_layer)
            self.channel_attention.append(channel_attention_layer_name)

            #spatial attention   
            spatial_attention_layer = self.make_spatial_attention_layers(plane)
            spatial_attention_layer_name = f'spatial_attention_layer{i + 2}'
            self.add_module(spatial_attention_layer_name, spatial_attention_layer)
            self.spatial_attention.append(spatial_attention_layer_name)
            #####################################
        self.softmax = nn.Softmax(dim=1)

        self._out_features = self._stage_names

        del self.vgg

    def make_attention_layers(self, inplanes, outplanes):
        inv_layers: List[nn.Module] = []
        conv2d = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, bias=False)
        inv_layers += [conv2d, nn.BatchNorm2d(outplanes)]
        inv_layers = nn.Sequential(*inv_layers) 
        return inv_layers
    
    def make_diff_layers(self, inplanes, outplanes):
        spc_layers: List[nn.Module] = []
        conv2d = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        spc_layers += [conv2d, nn.BatchNorm2d(outplanes), nn.ReLU(inplace=True)]
        spc_layers = nn.Sequential(*spc_layers) 
        return spc_layers
    
    def make_channel_attention_layers(self, planes, reduction=16):
        ca_layers: List[nn.Module] = []
        conv2d1 = nn.Conv2d(planes, planes // reduction, kernel_size=1, padding=0, bias=True)
        conv2d2 = nn.Conv2d(planes // reduction, planes, kernel_size=1, padding=0, bias=True)
        ca_layers += [nn.AdaptiveAvgPool2d(1), conv2d1, nn.ReLU(inplace=True), conv2d2, nn.Sigmoid()]
        ca_layers = nn.Sequential(*ca_layers) 
        return ca_layers

    def make_spatial_attention_layers(self, planes, reduction=16):
        sa_layers: List[nn.Module] = []
        conv2d1 = nn.Conv2d(planes, planes // reduction, kernel_size=7, padding=3, bias=True)
        conv2d2 = nn.Conv2d(planes // reduction, planes, kernel_size=7, padding=3, bias=True)
        sa_layers += [conv2d1, nn.ReLU(inplace=True), conv2d2, nn.Sigmoid()]
        sa_layers = nn.Sequential(*sa_layers)
        sa_layers = nn.Sequential(*sa_layers)
        return sa_layers

    def forward(self, x):
        features = {}
        #for name, stage in zip(self._stage_names, self.stages):
        #    x = stage(x)
        #    features[name] = x
        for i, (name, stage) in enumerate(zip(self._stage_names, self.stages)):
            layer_len = len(stage)
            for j in range(len(stage)):
                x = stage[j](x)
                if j == 1:
                    diff_feature = x
                if j == layer_len - 2:
                    if i >= 2:
                        inv_attention_layer = getattr(self, self.inv_attention[i - 2])
                        spc_layer = getattr(self, self.spc[i - 2])
                        spc_attention_layer = getattr(self, self.spc_attention[i - 2])
                        channel_attention_layer = getattr(self, self.channel_attention[i - 2])
                        spatial_attention_layer = getattr(self, self.spatial_attention[i - 2])

                        diff_feature = diff_feature - x
                        spc_feature = spc_layer(diff_feature)
                        spc_attention = spc_attention_layer(spc_feature)
                        inv_attention = inv_attention_layer(x)

                        weight = torch.cat((inv_attention.unsqueeze(1), spc_attention.unsqueeze(1)), 1)
                        weight_map = self.softmax(weight)
                        weight_map = torch.chunk(weight_map, 2, 1)
                        inv_attention = weight_map[0].squeeze(1)
                        spc_attention = weight_map[1].squeeze(1)

                        #######################
                        inv_feature = x
                        spec = spc_feature * spc_attention
                        #######################
                        x = spec + x * inv_attention
                        ca = channel_attention_layer(x)
                        x = x * ca
                        sa = spatial_attention_layer(x)
                        x = x * sa
            
            features[name] = x
        return features

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_backbone(cfg, _):
    return vgg_backbone(cfg)


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_fpn_backbone(cfg, _):
    # backbone = FPN(
    #     bottom_up=build_vgg_backbone(cfg),
    #     in_features=cfg.MODEL.FPN.IN_FEATURES,
    #     out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
    #     norm=cfg.MODEL.FPN.NORM,
    #     top_block=LastLevelMaxPool(),
    # )

    bottom_up = vgg_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        # fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    # return backbone

    return backbone
