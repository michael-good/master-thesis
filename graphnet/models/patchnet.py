import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PatchNet(nn.Module):
    
    def __init__(self, out_dim, pretrained=True):
        super(PatchNet, self).__init__()
        self.pretrained = pretrained
        self.model = self._modify_model(out_dim)
            
    def _modify_model(self, out_dim):
        model = models.vgg16_bn(pretrained=self.pretrained)
        model.features = nn.Sequential(*[model.features[i] for i in range(33)])
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Sequential(*[nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, out_dim)])
        return model

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        x = F.normalize(x, p=2, dim=1)
        return x  
            
