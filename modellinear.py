import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

#def conv3x3(in_channels, out_channels, **kwargs):
#    return MetaSequential(
#        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
#        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
#        nn.ReLU(),
#        nn.MaxPool2d(2)
 #   )

class LinearNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(LinearNeuralNetwork, self).__init__()
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            MetaLinear(in_channels, hidden_size*4),
            MetaLinear(hidden_size*4, hidden_size*2),
            MetaLinear(hidden_size*2, hidden_size),
            MetaLinear(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
       # print(features.size)
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits
