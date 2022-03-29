import torch

from luffy.models.efficientnet import EfficientNetB0


def test_efficientnet():
    model = EfficientNetB0()

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
