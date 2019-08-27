import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG_rebuild', 'vgg11_rebuild', 'vgg11_bn_rebuild', 'vgg13_rebuild', 'vgg13_bn_rebuild', 'vgg16_rebuild', 'vgg16_bn_rebuild',
    'vgg19_bn_rebuild', 'vgg19_rebuild',
]


class VGG_rebuild(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_rebuild, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Linear(512, 512 * 7 * 7),
            nn.Linear(prune_num_0_3[3], 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x1 = self.features(x)
        x = x1.view(x1.size(0), -1)
        x = self.classifier(x)
        return x, x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, cfg1, batch_norm=False):
    layers = []
    in_channels = 3
    i = 0
    j = 0
    for v in cfg:
        #print('v: {}, f: {}'.format(v, f))
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            j = j + 1
            in_channels = cfg1[j]
            #j = j + 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            #conv2d = nn.Conv2d(f_num[i], in_channels, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
#            if v != in_channels:
#                i = i + 1
            in_channels = v
            j = j + 1

    return nn.Sequential(*layers)

#f_num = [45, 45, 128, 256, 512]
prune_num_0_1 = [58,116,231,461]
prune_num_0_2 = [52,103,205,410]
prune_num_0_3 = [45,90,180,359]
prune_num_0_4 = [39,77,154,308]
prune_num_0_5 = [32,64,128,256]

rate=0.7

if rate ==0.9:
# 10%
    cfg = {
    'A': [58, 'M', 116, 'M', 231, 231, 'M', 461, 461, 'M', 461, 461, 'M'],
    'B': [58, 58, 'M', 116, 116, 'M', 231, 231, 'M', 461, 461, 'M', 461, 461, 'M'],
    'D': [58, 58, 'M', 116, 116, 'M', 231, 231, 231, 'M', 461, 461, 461, 'M', 461, 461, 461, 'M'],
    'E': [58, 58, 'M', 116, 116, 'M', 231, 231, 231, 231, 'M', 461, 461, 461, 461, 'M', 461, 461, 461, 461, 'M'],
}
    cfg1 = {
    'A1': [64, 'M', 58, 'M', 116, 116, 'M', 231, 231, 'M', 461, 461, 'M', 461],
    'B1': [64, 64, 'M', 58, 58, 'M', 116, 116, 'M', 231, 231, 'M', 461, 461, 'M', 461],
    'D1': [64, 64, 'M', 58, 58, 'M', 116, 116, 116, 'M', 231, 231, 231, 'M', 461, 461, 461, 'M', 461],
    'E1': [64, 64, 'M', 58, 58, 'M', 116, 116, 116, 116, 'M', 231, 231, 231, 231, 'M', 461, 461, 461, 461, 'M', 461],
}
elif rate ==0.8:
# 20%
    cfg = {
    'A': [52, 'M', 103, 'M', 205, 205, 'M', 410, 410, 'M', 410, 410, 'M'],
    'B': [52, 52, 'M', 103, 103, 'M', 205, 205, 'M', 410, 410, 'M', 410, 410, 'M'],
    'D': [52, 52, 'M', 103, 103, 'M', 205, 205, 205, 'M', 410, 410, 410, 'M', 410, 410, 410, 'M'],
    'E': [52, 52, 'M', 103, 103, 'M', 205, 205, 205, 205, 'M', 410, 410, 410, 410, 'M', 410, 410, 410, 410, 'M'],
}
    cfg1 = {
    'A1': [64, 'M', 52, 'M', 103, 103, 'M', 205, 205, 'M', 410, 410, 'M', 410],
    'B1': [64, 64, 'M', 52, 52, 'M', 103, 103, 'M', 205, 205, 'M', 410, 410, 'M', 410],
    'D1': [64, 64, 'M', 52, 52, 'M', 103, 103, 103, 'M', 205, 205, 205, 'M', 410, 410, 410, 'M', 410],
    'E1': [64, 64, 'M', 52, 52, 'M', 103, 103, 103, 103, 'M', 205, 205, 205, 205, 'M', 410, 410, 410, 410, 'M', 410],
}
elif rate ==0.7:
# 30%
    cfg = {
    'A': [45, 'M', 90, 'M', 180, 180, 'M', 359, 359, 'M', 359, 359, 'M'],
    'B': [45, 45, 'M', 90, 90, 'M', 180, 180, 'M', 359, 359, 'M', 359, 359, 'M'],
    'D': [45, 45, 'M', 90, 90, 'M', 180, 180, 180, 'M', 359, 359, 359, 'M', 359, 359, 359, 'M'],
#    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [45, 45, 'M', 90, 90, 'M', 180, 180, 180, 180, 'M', 359, 359, 359, 359, 'M', 359, 359, 359, 359, 'M'],
}

    cfg1 = {
    'A1': [64, 'M', 45, 'M', 90, 90, 'M', 180, 180, 'M', 359, 359, 'M', 359],
    'B1': [64, 64, 'M', 45, 45, 'M', 90, 90, 'M', 180, 180, 'M', 359, 359, 'M', 359],
    'D1': [64, 64, 'M', 45, 45, 'M', 90, 90, 90, 'M', 180, 180, 180, 'M', 359, 359, 359, 'M', 359],
    'E1': [64, 64, 'M', 45, 45, 'M', 90, 90, 90, 90, 'M', 180, 180, 180, 180, 'M', 359, 359, 359, 359, 'M', 359],
}

# 40%
# cfg = {
    # 'A': [39, 'M', 77, 'M', 154, 154, 'M', 308, 308, 'M', 308, 308, 'M'],
    # 'B': [39, 39, 'M', 77, 77, 'M', 154, 154, 'M', 308, 308, 'M', 308, 308, 'M'],
    # 'D': [39, 39, 'M', 77, 77, 'M', 154, 154, 154, 'M', 308, 308, 308, 'M', 308, 308, 308, 'M'],
    # 'E': [39, 39, 'M', 77, 77, 'M', 154, 154, 154, 154, 'M', 308, 308, 308, 308, 'M', 308, 308, 308, 308, 'M'],
# }

# 50%
# cfg = {
    # 'A': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    # 'B': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    # 'D': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    # 'E': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
# }

def vgg11_rebuild(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['A']), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn_rebuild(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['A'], cfg1['A1'], batch_norm=True), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13_rebuild(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['B']), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn_rebuild(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['B'], cfg1['B1'], batch_norm=True), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16_rebuild(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['D']), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn_rebuild(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['D'], cfg1['D1'], batch_norm=True), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19_rebuild(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['E']), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn_rebuild(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_rebuild(make_layers(cfg['E'], cfg1['E1'], batch_norm=True), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
