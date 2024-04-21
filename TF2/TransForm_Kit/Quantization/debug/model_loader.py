# Copyright 2019 Inspur Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.parallel
from models.ResNet50 import ResNet50
from models.SqueezeNet import SqueezeNet
from models.GoogLeNet import GoogLeNet
from models.SSD import SSD
from net_structure import structure_hook

def load_model(net_name):
    if net_name == 'squeezenet':
        net = SqueezeNet.SqueezeNet('1_1')
    if net_name == 'googlenet':
        net = GoogLeNet.GoogLeNet()
    if net_name == 'resnet50':
        net = ResNet50.resnet50()
        net = torch.nn.DataParallel(net).cuda()
    if net_name == 'ssd':
        net = SSD.build_ssd('test', 300, 21)
    pars_path = '../weights/' + net_name + '.pth.tar'
    feature_path = 'Feature_float/' + net_name
    net.eval()
    checkpoint = torch.load(pars_path)
    net.load_state_dict(checkpoint['state_dict'])
    layer_name, Features = structure_hook(net,net_name)
    return net, layer_name, Features, feature_path

