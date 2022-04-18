import torch

from luffy.models.vision_permutator import *


def test_vision_permutator_s14():
    model = ViPS14(image_size=224, num_classes=1000)

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
