import torch

from luffy.models.vit import ViTB16


def test_vit():
    model = ViTB16(image_size=224, num_classes=1000)

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
