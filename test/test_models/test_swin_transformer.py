import torch

from luffy.models.swin_transformer import *


def test_swin_t():
    model = SwinT(image_size=224, num_classes=1000)

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
