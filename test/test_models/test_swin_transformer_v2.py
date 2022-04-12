import torch

from luffy.models.swin_transformer_v2 import *


def test_swinv2_t():
    model = SwinV2T(image_size=224, num_classes=1000)

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
