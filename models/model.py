import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes=1000, pretrained=False, metric='linear'):
        super().__init__()
        self.model_name = model_name

        if model_name == 'res18':
            backbone = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])
            plane = 512
        elif model_name == 'res34':
            backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-2])
            plane = 512
        elif model_name == 'res50':
            backbone = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
            plane = 2048
        elif model_name == 'resx50':
            backbone = nn.Sequential(*list(models.resnext50_32x4d(pretrained=pretrained).children())[:-2])
            plane = 2048
        else:
            raise ValueError('model - {} is not support'.format(model_name))

        self.backbone = backbone

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if metric == 'linear':
            self.metric = nn.Linear(plane, num_classes)
        else:
            self.metric = None

    def forward(self, x):
        feat = self.backbone(x)
        feat_flat = self.pool(feat).view(feat.size(0), -1)
        out = self.metric(feat_flat)
        if self.training:
            return out, None
        else:
            return out


if __name__ == '__main__':
    model = BaseModel(model_name='res50').eval()
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    print(out.size())
    print(model)