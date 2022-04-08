import torch

from luffy.models.vision_transformer import *


def test_vit_b16():
    model = ViTB16(image_size=224, num_classes=1000)

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
