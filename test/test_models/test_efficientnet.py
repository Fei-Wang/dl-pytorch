import torch

from luffy.models.efficientnet import *


def test_efficientnet_b0():
    model = EfficientNetB0()

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
